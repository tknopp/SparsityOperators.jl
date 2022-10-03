export NormalOp, normalOperator

struct NormalOp{S,D,V} 
  parent::S
  weights::D
  tmp::V
end

function NormalOp(parent, weights)
  T = promote_type(eltype(parent), eltype(weights))
  tmp = Vector{T}(undef, size(parent, 1))
  return NormalOp(parent, weights, tmp)
end

function Base.copy(S::NormalOp)
  return NormalOp(copy(S.parent), S.weights, copy(S.tmp))
end

function normalOperator(parent, weights=opEye(eltype(parent), size(parent,1)))
  return NormalOp(parent, weights)
end

function Base.size(S::NormalOp)
  return (S.parent.ncol, S.parent.ncol)
end

function Base.size(S::NormalOp, dim)
  if dim == 1 || dim == 2
    return S.parent.ncol
  else
    error()
  end
end

function LinearAlgebra.mul!(y, S::NormalOp, x)
  mul!(S.tmp, S.parent, x)
  mul!(S.tmp, S.weights, S.tmp) # This can be dangerous. We might need to create two tmp vectors
  return mul!(y, adjoint(S.parent), S.tmp)
end

# Generic fallback -> TODO avoid allocations
function Base.:*(N::NormalOp, x::AbstractVector)
  y = similar(x)
  mul!(y,N,x)
  return y
end
