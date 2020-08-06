export NormalOp, normalOperator

struct NormalOp{S,D} 
  parent::S
  weights::D
end

function Base.copy(S::NormalOp)
  return NormalOp(copy(S.parent), S.weights)
end

function normalOperator(S, W=I)
  return NormalOp(S,W)
end

# Generic fallback -> TODO avoid allocations
function Base.:*(N::NormalOp, x::AbstractVector)

  #@info size(x)  size(N.parent)   size(N.weights)   

  d = adjoint(N.parent)*(N.weights*(N.parent*x))
  return d
end