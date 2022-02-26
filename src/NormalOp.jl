export NormalOp, normalOperator

struct NormalOp{S,D} 
  parent::S
  weights::D
end

function Base.copy(S::NormalOp)
  return NormalOp(copy(S.parent), S.weights)
end

function normalOperator(S, W=opEye())
  return NormalOp(S,W)
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

function LinearAlgebra.mul!(x, S::NormalOp, b)
  x .= S * b
  return x
end

# Generic fallback -> TODO avoid allocations
function Base.:*(N::NormalOp, x::AbstractVector)

  #@info size(x)  size(N.parent)   size(N.weights)   

  d = adjoint(N.parent)*(N.weights*(N.parent*x))
  return d
end