export FFTOp
import Base.copy

mutable struct FFTOp <: AbstractLinearOperator{ComplexF64}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod :: Function
  tprod :: Nothing
  ctprod :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  plan
  iplan
  shift::Bool
  unitary::Bool
  T::Type
end

"""
  FFTOp(T::Type, shape::Tuple, shift=true, unitary=true)

returns an operator which performs an FFT on Arrays of type T

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* (`shift=true`)  - if true, fftshifts are performed
* (`unitary=true`)  - if true, FFT is normalized such that it is unitary
"""
function FFTOp(T::Type, shape::Tuple, shift=true; unitary=true)
  plan = plan_fft(zeros(T, shape);flags=FFTW.MEASURE)
  iplan = plan_ifft(zeros(T, shape);flags=FFTW.MEASURE)
  if unitary
    facF = 1.0/sqrt(prod(shape))
    facB = sqrt(prod(shape))
  else
    facF = 1.0
    facB = prod(shape)
  end

  if shift
    return FFTOp(prod(shape), prod(shape), false, false
              , x->vec(fftshift(plan*fftshift(reshape(x,shape))))*facF
              , nothing
              , y->vec(ifftshift(iplan*ifftshift(reshape(y,shape))))*facB
              , 0, 0, 0
              , plan
              , iplan
              , shift
              , unitary
              , T)
  else
    return FFTOp(prod(shape), prod(shape), false, false
            , x->vec(plan*(reshape(x,shape)))*facF
            , nothing
            , y->vec(iplan*(reshape(y,shape)))*facB 
            , 0, 0, 0
            , plan
            , iplan
            , shift
            , unitary
            , T)
  end
end

function Base.copy(S::FFTOp)
  return FFTOp(S.T, size(S.plan), S.shift, unitary=S.unitary)
end