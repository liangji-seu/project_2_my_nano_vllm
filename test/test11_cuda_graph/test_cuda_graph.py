import pytest
import torch

from my_vllm.attention.metadata import (
    FullAttentionMetadata,
    FullAttentionMetadataCollection,
)
from my_vllm.compilation.cuda_graph import (
    BatchDescriptor,
    CUDAGraphDispatcher,
    CUDAGraphMode,
    CUDAGraphWrapper,
)
from my_vllm.config import EngineConfig, make_arg_parser
from my_vllm.worker.gpu_model_runner import ModelForwardInputs


def make_decode_inputs(device: torch.device, num_reqs: int = 2):
    input_ids = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    positions = torch.arange(8, 8 + num_reqs, dtype=torch.int64, device=device)
    query_start_loc = torch.arange(
        num_reqs + 1, dtype=torch.int32, device=device
    )
    seq_lens = positions.to(torch.int32) + 1
    metadata = FullAttentionMetadata(
        kv_cache_group_id=0,
        layer_names=("layer",),
        block_size=16,
        causal=True,
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=256,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_computed_tokens=seq_lens - 1,
        num_scheduled_tokens=torch.ones(
            num_reqs, dtype=torch.int32, device=device
        ),
        positions=positions,
        block_table=torch.zeros((num_reqs, 2), dtype=torch.int32, device=device),
        slot_mapping=torch.arange(num_reqs, dtype=torch.int64, device=device),
    )
    collection = FullAttentionMetadataCollection(
        by_group={0: metadata}, by_layer={"layer": metadata}
    )
    return ModelForwardInputs(
        input_ids=input_ids,
        inputs_embeds=None,
        positions=positions,
        attention_metadata=collection,
        logits_indices=torch.arange(num_reqs, dtype=torch.int64, device=device),
        is_decode=True,
    )


def test_cuda_graph_cli_defaults_and_disable_flag():
    parser = make_arg_parser()
    defaults = EngineConfig.from_cli_args(parser.parse_args([]))
    disabled = EngineConfig.from_cli_args(
        parser.parse_args(
            [
                "--disable-cuda-graph",
                "--cuda-graph-seq-len-bucket-size",
                "128",
                "--cuda-graph-num-warmups",
                "2",
            ]
        )
    )

    assert defaults.enable_cuda_graph is True
    assert defaults.cuda_graph_seq_len_bucket_size == 256
    assert defaults.cuda_graph_num_warmups == 1
    assert defaults.cuda_graph_capture_sizes == (1, 2, 4, 8, 16, 32)
    assert disabled.enable_cuda_graph is False
    assert disabled.cuda_graph_seq_len_bucket_size == 128
    assert disabled.cuda_graph_num_warmups == 2


def test_dispatcher_builds_independent_valid_mode_key_library():
    dispatcher = CUDAGraphDispatcher(
        enabled=True,
        max_num_reqs=8,
        max_model_len=4096,
        capture_batch_sizes=(1, 2, 4, 8, 16),
        seq_len_bucket_size=256,
    )
    assert not dispatcher.is_initialized
    assert not dispatcher.valid_keys[CUDAGraphMode.FULL]

    dispatcher.initialize_cudagraph_keys()
    assert dispatcher.capture_batch_sizes == (1, 2, 4, 8)
    assert len(dispatcher.valid_keys[CUDAGraphMode.FULL]) == 20

    mode, descriptor = dispatcher.dispatch(
        num_tokens=2,
        num_reqs=2,
        max_seq_len=700,
        is_uniform_decode=True,
    )
    assert mode is CUDAGraphMode.FULL
    assert descriptor == BatchDescriptor(2, 2, 1024, True)

    # 未配置的 batch size 和非纯 Decode 不允许偷偷惰性建图。
    assert dispatcher.dispatch(
        num_tokens=3,
        num_reqs=3,
        max_seq_len=700,
        is_uniform_decode=True,
    ) == (CUDAGraphMode.NONE, None)
    assert dispatcher.dispatch(
        num_tokens=4,
        num_reqs=2,
        max_seq_len=700,
        is_uniform_decode=False,
    ) == (CUDAGraphMode.NONE, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA Graph")
def test_wrapper_active_capture_then_runtime_replay():
    device = torch.device("cuda:0")
    inputs = make_decode_inputs(device)

    def runnable(model_inputs):
        return model_inputs.input_ids.to(torch.float32) * 2 + model_inputs.positions

    descriptor = BatchDescriptor(
        num_tokens=2, num_reqs=2, max_seq_len=256, is_uniform=True
    )
    wrapper = CUDAGraphWrapper(runnable, device=device)
    # 模拟 GPUModelRunner.capture_model()：只有显式 capturing=True 才能建图。
    first = wrapper(
        inputs,
        mode=CUDAGraphMode.FULL,
        descriptor=descriptor,
        is_graph_capturing=True,
    ).clone()
    assert wrapper.capture_count == 1
    assert wrapper.replay_count == 0
    torch.testing.assert_close(first, torch.tensor([10.0, 13.0], device=device))

    wrapper.finish_capture()
    inputs.input_ids.add_(10)
    inputs.positions.add_(20)
    second = wrapper(
        inputs, mode=CUDAGraphMode.FULL, descriptor=descriptor
    ).clone()
    assert wrapper.capture_count == 1
    assert wrapper.replay_count == 1
    torch.testing.assert_close(second, torch.tensor([50.0, 53.0], device=device))
