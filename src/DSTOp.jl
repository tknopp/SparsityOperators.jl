export DSTOp

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
  return LinearOperator{T}(prod(shape), prod(shape), true, false
            , x->vec(plan*reshape(x,shape)).*weights(shape, T)
            , nothing
            , y->vec(iplan*reshape(y ./ weights(shape, T) ,shape)) ./ (8*prod(shape))  )
end

function weights(s, T::Type)
  w = ones(T,s...)./T(sqrt(8*prod(s)))
  w[s[1],:,:]./= T(sqrt(2))
  w[:,s[2],:]./= T(sqrt(2))
  w[:,:,s[3]]./= T(sqrt(2))
  return reshape(w,prod(s))
end
