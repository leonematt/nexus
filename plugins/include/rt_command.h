#ifndef RT_COMMAND_H
#define RT_COMMAND_H

#include <array>

#include <rt_buffer.h>

#define RT_COMMAND_MAX_ARGS 64

namespace nxs {
namespace rt {

template <typename Tkernel, typename Tevent, typename Tstream>
class Command {
 protected:
  Tkernel kernel;
  Tevent event;
  nxs_command_type type;
  nxs_int event_value;
  float time_ms;
  nxs_uint settings;
  std::array<void *, RT_COMMAND_MAX_ARGS> args;
  std::array<void *, RT_COMMAND_MAX_ARGS> args_ref;
  int args_count;
  nxs_dim3 block_size;
  nxs_dim3 grid_size;
  nxs_uint shared_memory_size;

 public:
  Command(Tkernel kernel, nxs_uint settings = 0)
      : kernel(kernel),
        type(NXS_CommandType_Dispatch),
        time_ms(0),
        settings(settings) {
          args_ref.fill(nullptr);
          args_count = 0;
        }

        Command(Tevent event, nxs_command_type type, nxs_int event_value = 1,
                nxs_uint settings = 0)
            : event(event),
              type(type),
              event_value(event_value),
              settings(settings) {
          args_count = 0;
        }

  virtual ~Command() = default;

  nxs_command_type getType() const { return type; }

  float getTime() const { return time_ms; }

  int getArgsCount() const { return args_count; }

  nxs_status setArgument(nxs_int argument_index, nxs::rt::Buffer *buffer) {
    if (argument_index >= RT_COMMAND_MAX_ARGS) return NXS_InvalidArgIndex;
    args_count = std::max(args_count, argument_index + 1);

    args[argument_index] = buffer->get();
    args_ref[argument_index] = &args[argument_index];
    return NXS_Success;
  }

  nxs_status setScalar(nxs_int argument_index, void *value) {
    if (argument_index >= RT_COMMAND_MAX_ARGS) return NXS_InvalidArgIndex;
    args_count = std::max(args_count, argument_index + 1);

    // Object owned by Core API
    args[argument_index] = value;
    args_ref[argument_index] = value;
    return NXS_Success;
  }

  nxs_status finalize(nxs_dim3 grid_size, nxs_dim3 block_size, nxs_uint shared_memory_size) {
    // TODO: check if all arguments are valid
    this->grid_size = grid_size;
    this->block_size = block_size;
    this->shared_memory_size = shared_memory_size;
    return NXS_Success;
  }

  virtual nxs_status runCommand(Tstream stream) = 0;

  virtual void release() {}
};

}  // namespace rt
}  // namespace nxs

#endif  // RT_COMMAND_H