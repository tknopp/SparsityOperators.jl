export DCTOp

mutable struct DCTOp{T} <: AbstractLinearOperator{T}
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
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  plan
  dcttype::Int
end

"""
  DCTOp(T::Type, shape::Tuple, dcttype=2)

returns a `DCTOp <: AbstractLinearOperator` which performs a DCT on a given input array.

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* `dcttype`       - type of DCT (currently `2` and `4` are supported)
"""
function DCTOp(T::Type, shape::Tuple, dcttype=2)

  function dct_multiply(res, plan, x, α, β::T2, shape, T::Type, factor) where {T2}
    if β == zero(T2)
      res .= (factor*α) .* vec(plan*reshape(x,shape))
    else
      res .= (factor*α) .* vec(plan*reshape(x,shape)) .+ β .* q
    end
  end

  if dcttype == 2
    plan = plan_dct(zeros(T,shape))
    iplan = plan_idct(zeros(T,shape))

    prod! = (res, x, α, β)  -> dct_multiply(res, plan, x, α, β, shape, T, one(T))
    tprod! = (res, x, α, β)  -> dct_multiply(res, iplan, x, α, β, shape, T, one(T))

  elseif dcttype == 4
    factor = T(sqrt(1.0/(prod(shape)* 2^length(shape)) ))
    plan = FFTW.plan_r2r(zeros(T,shape),FFTW.REDFT11)
    prod! = (res, x, α, β) -> dct_multiply(res, plan, x, α, β, shape, T, factor)
    tprod! = (res, x, α, β) -> dct_multiply(res, plan, x, α, β, shape, T, factor)
  else
    error("DCT type $(dcttype) not supported")
  end

  return DCTOp{T}(prod(shape), prod(shape), false, false,
                      prod!, nothing, tprod!,
                      0, 0, 0, true, true, true,
                      plan, dcttype)
end

function Base.copy(S::DCTOp)
  return DCTOp(eltype(S), size(S.plan), S.dcttype)
end