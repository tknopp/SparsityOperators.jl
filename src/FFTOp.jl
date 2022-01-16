export FFTOp
import Base.copy

mutable struct FFTOp{T} <: AbstractLinearOperator{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  plan
  iplan
  shift::Bool
  unitary::Bool
end

LinearOperators.has_args5(op::FFTOp) = true
LinearOperators.use_prod5!(op::FFTOp) = true
LinearOperators.isallocated5(op::FFTOp) = true

"""
  FFTOp(T::Type, shape::Tuple, shift=true, unitary=true)

returns an operator which performs an FFT on Arrays of type T

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* (`shift=true`)  - if true, fftshifts are performed
* (`unitary=true`)  - if true, FFT is normalized such that it is unitary
"""
function FFTOp(T::Type, shape::Tuple, shift=true; unitary=true, cuda::Bool=false)
  if cuda
    @assert CUDA.functional()==true "a functional CUDA setup is required when using the option `cuda=true` in FFTOp"
    plan = plan_fft(CuArray{T}(undef,shape))
    iplan = plan_ifft(CuArray{T}(undef,shape))
  else
    plan = plan_fft(zeros(T, shape);flags=FFTW.MEASURE)
    iplan = plan_ifft(zeros(T, shape);flags=FFTW.MEASURE)
  end
  if unitary
    facF = T(1.0/sqrt(prod(shape)))
    facB = T(sqrt(prod(shape)))
  else
    facF = T(1.0)
    facB = T(prod(shape))
  end

  function fft_multiply(res, plan, x, α, β::T2, shape, T::Type, factor) where {T2}
    if β == zero(T2)
      res .= (factor*α) .* vec(plan*reshape(x,shape))
    else
      res .= (factor*α) .* vec(plan*reshape(x,shape)) .+ β .* q
    end
  end

  function fft_multiply_shift(res, plan, x, α, β::T2, shape, T::Type, factor) where {T2}
    if β == zero(T2)
      res .= (factor*α) .* vec(fftshift(plan*fftshift(reshape(x,shape))))
    else
      res .= (factor*α) .* vec(fftshift(plan*fftshift(reshape(x,shape)))) .+ β .* q
    end
  end

  function fft_multiply_ishift(res, plan, x, α, β::T2, shape, T::Type, factor) where {T2}
    if β == zero(T2)
      res .= (factor*α) .* vec(ifftshift(iplan*ifftshift(reshape(x,shape))))
    else
      res .= (factor*α) .* vec(ifftshift(iplan*ifftshift(reshape(x,shape)))) .+ β .* q
    end
  end

  if shift
    return FFTOp{T}(prod(shape), prod(shape), false, false
              , (res, x, α, β) -> fft_multiply_shift(res, plan, x, α, β, shape, T, facF) 
              , nothing
              , (res, x, α, β) -> fft_multiply_ishift(res, iplan, x, α, β, shape, T, facB) 
              , 0, 0, 0
              , plan
              , iplan
              , shift
              , unitary)
  else
    return FFTOp{T}(prod(shape), prod(shape), false, false
            , (res, x, α, β) -> fft_multiply(res, plan, x, α, β, shape, T, facF) 
            , nothing
            , (res, x, α, β) -> fft_multiply(res, iplan, x, α, β, shape, T, facB)
            , 0, 0, 0
            , plan
            , iplan
            , shift
            , unitary)
  end
end

function Base.copy(S::FFTOp)
  return FFTOp(eltype(S), size(S.plan), S.shift, unitary=S.unitary)
end