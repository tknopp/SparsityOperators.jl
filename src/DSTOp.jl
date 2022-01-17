export DSTOp

mutable struct DSTOp{T} <: AbstractLinearOperator{T}
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
end

LinearOperators.has_args5(op::DSTOp) = true
LinearOperators.use_prod5!(op::DSTOp) = true
LinearOperators.isallocated5(op::DSTOp) = true

"""
  DSTOp(T::Type, shape::Tuple)

returns a `LinearOperator` which performs a DST on a given input array.

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
"""
function DSTOp(T::Type, shape::Tuple)
  plan = FFTW.plan_r2r(zeros(T,shape),FFTW.RODFT10)
  iplan = FFTW.plan_r2r(zeros(T,shape),FFTW.RODFT01)
  return DSTOp{T}(prod(shape), prod(shape), true, false
            , wrapProd(x->T.( vec(plan*reshape(x,shape)) ).*weights(shape, T))
            , nothing
            , wrapProd(y->T.( vec(iplan*reshape(y ./ weights(shape, T) ,shape)) ./ (8*prod(shape)) ))
            , 0, 0, 0
            , plan
            , iplan)
end

function weights(s, T::Type)
  w = ones(T,s...)./T(sqrt(8*prod(s)))
  w[s[1],:,:]./= T(sqrt(2))
  if length(s)>1
    w[:,s[2],:]./= T(sqrt(2))
    if length(s)>2
      w[:,:,s[3]]./= T(sqrt(2))
    end
  end
  return reshape(w,prod(s))
end


function Base.copy(S::DSTOp)
  return DSTOp(eltype(S), size(S.plan))
end