// ABI shim for running prebuilt flash_attn (built for torch <= 2.9) on torch 2.11.
//
// torch 2.11 changed the line-number parameter of
//   c10::cuda::c10_cuda_check_implementation(int, const char*, const char*, <int|unsigned int>, bool)
// from `int` to `unsigned int`, so flash_attn's prebuilt .so fails to load with:
//   undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
//
// This shim defines the missing `int` overload and forwards to torch's `unsigned int` one.
// Build:
//   TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
//   g++ -O2 -fPIC -shared -o libabishim.so abi_shim.cpp -L"$TORCH_LIB" -lc10_cuda -Wl,-rpath,"$TORCH_LIB"
// Use:
//   LD_PRELOAD=/workspace/exploration/libabishim.so <launch...>
// (also wired into the Ray worker runtime env in verl/trainer/constants_ppo.py)
//
// See REPRODUCE.md §8.3.

namespace c10 { namespace cuda {
  // Provided by torch 2.11's libc10_cuda.so (unsigned int line-number param).
  void c10_cuda_check_implementation(int, const char*, const char*, unsigned int, bool);
  // The symbol flash_attn (built for torch <= 2.9) expects (int line-number param).
  void c10_cuda_check_implementation(int a, const char* b, const char* c, int d, bool e) {
    c10_cuda_check_implementation(a, b, c, static_cast<unsigned int>(d), e);
  }
}}
