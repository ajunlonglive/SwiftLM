// Copyright © 2026 SharpAI
// moe_stream_op.cpp
// Custom MLX Operation that combines GatherMM with SSD Streaming

#include "mlx/core/moe_stream_op.h"
#include <iostream>
#include <chrono>
#include <atomic>
#include "mlx/primitives.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

// Static SSD metric trackers for aggregate logging
static std::atomic<size_t> g_total_bytes_read{0};
static std::atomic<uint64_t> g_total_read_ns{0};
static std::atomic<size_t> g_read_count{0};
static std::atomic<uint64_t> g_last_log_ns{0};

namespace mlx::core {

class LoadSSDExpert : public Primitive {
public:
    LoadSSDExpert(
        Stream s,
        uint32_t active_expert,
        std::shared_ptr<fast::SSDStreamer> streamer,
        const std::vector<off_t>& expert_offsets
    ) : Primitive(s), active_expert_(active_expert), streamer_(streamer), expert_offsets_(expert_offsets) {}
    
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        eval_impl(inputs, outputs);
        
        auto& d = metal::device(mlx::core::Device::gpu);
        d.add_temporary(outputs[0], stream().index);
    }
    
    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        eval_impl(inputs, outputs);
    }
    
    void eval_impl(const std::vector<array>& inputs, std::vector<array>& outputs) {
        auto& o = outputs[0];
        
        uint32_t active_expert = active_expert_;
        if (active_expert + 1 >= expert_offsets_.size()) {
            throw std::runtime_error("[LoadSSDExpert] Expert index out of bounds.");
        }
        
        off_t block_offset = expert_offsets_[active_expert];
        size_t matrix_bytes = static_cast<size_t>(expert_offsets_[active_expert + 1] - block_offset);

        // We use MLX's allocator to get Metal-accessible (unified) memory.
        o.set_data(allocator::malloc(matrix_bytes));

        auto start_read = std::chrono::high_resolution_clock::now();
        streamer_->load_sync(block_offset, matrix_bytes, o.data<void>());
        auto end_read = std::chrono::high_resolution_clock::now();

        // ─────────────────────────────────────────────────────────────────────
        // AGGREGATE LOGGING — 1-second metric intervals
        // ─────────────────────────────────────────────────────────────────────
        g_total_bytes_read += matrix_bytes;
        g_total_read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_read - start_read).count();
        g_read_count++;

        auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t last = g_last_log_ns.load();
        
        if (now_ns - last >= 1000000000ULL) {
            if (g_last_log_ns.compare_exchange_strong(last, now_ns)) {
                size_t count = g_read_count.exchange(0);
                size_t bytes = g_total_bytes_read.exchange(0);
                uint64_t ns_time = g_total_read_ns.exchange(0);
                if (count > 0) {
                    double avg_ms = (ns_time / 1000000.0) / count;
                    double mb = bytes / (1024.0 * 1024.0);
                    std::cout << "[⚡️ SSD Stream] " << mb << " MB/s over " 
                              << count << " chunks | Avg latency per chunk: " 
                              << avg_ms << " ms" << std::endl;
                }
            }
        }
    }
    
    std::vector<array> vjp(
        const std::vector<array>& inputs,
        const std::vector<array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<array>& outputs) override {
        throw std::runtime_error("[LoadSSDExpert] backward pass (VJP) is unsupported.");
    }

    std::vector<array> jvp(
        const std::vector<array>& inputs,
        const std::vector<array>& tangents,
        const std::vector<int>& argnums) override {
        throw std::runtime_error("[LoadSSDExpert] backward pass (JVP) is unsupported.");
    }
    
    bool is_equivalent(const Primitive& other) const override {
        return false;
    }
    
    const char* name() const override {
        return "LoadSSDExpert";
    }
    
private:
    uint32_t active_expert_;
    std::shared_ptr<fast::SSDStreamer> streamer_;
    std::vector<off_t> expert_offsets_;
};

MLX_API array streamed_gather_mm(
    const array& x, // Ignored logic-wise, kept for ABI signature mapping in fast.cpp
    const array& w_shape,
    uint32_t active_expert,
    std::shared_ptr<fast::SSDStreamer> streamer,
    const std::vector<off_t>& expert_offsets,
    StreamOrDevice s
) {
    // Output shape: [1, outputDims, inputDims]
    auto OD = w_shape.shape(1);
    auto ID = w_shape.shape(2);
    
    return array(
        {1, static_cast<int>(OD), static_cast<int>(ID)}, uint32,
        std::make_unique<LoadSSDExpert>(to_stream(s), active_expert, streamer, expert_offsets),
        {x} // MUST pass a dummy input or the node gets eliminated!
    );
}

} // namespace mlx::core
