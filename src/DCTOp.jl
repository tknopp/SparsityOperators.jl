export DCTOp

mutable struct DCTOp{T} <: AbstractLinearOperator{T}
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
  dcttype::Int64
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
  if dcttype == 2
    plan = plan_dct(zeros(T,shape))
    iplan = plan_idct(zeros(T,shape))
    return DCTOp{T}(prod(shape), prod(shape), true, false
            , x->vec(plan*reshape(x,shape))
            , nothing
            , y->vec(iplan*reshape(y,shape))
            , 0, 0, 0
            , plan
            , iplan
            , dcttype)
  elseif dcttype == 4
    factor = sqrt(1.0/(prod(shape)* 2^length(shape)) )
    plan = FFTW.plan_r2r(zeros(T,shape),FFTW.REDFT11)
    # return DCTOp{T}(prod(shape), prod(shape), true, false
    #         , x->vec((plan*reshape(x,shape)).*factor)
    #         , x->vec((plan*reshape(x,shape)).*factor)
    #         , nothing
    #         , 0, 0, 0
    #         , plan
    #         , plan
    #         , dcttype)
    return LinearOperator{T}(prod(shape), prod(shape), true, false
            , x->T.( vec((plan*reshape(x,shape)).*factor) )
            , x->T.( vec((plan*reshape(x,shape)).*factor) )
            , nothing )
  else
    error("DCT type $(dcttype) not supported")
  end
end

function Base.copy(S::DCTOp)
  return DCTOp(eltype(S), size(S.plan), S.dcttype)
end
