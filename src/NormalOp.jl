export NormalOp, normalOperator

struct NormalOp{S,D,V} 
  parent::S
  weights::D
  tmp::V
end

function Base.copy(S::NormalOp)
  return NormalOp(copy(S.parent), S.weights, copy(S.tmp))
end

function normalOperator(S, W=opEye(eltype(S), size(S,1)))
  tmp = Vector{eltype(S)}(undef, size(S, 1))
  return NormalOp(S,W,tmp)
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
