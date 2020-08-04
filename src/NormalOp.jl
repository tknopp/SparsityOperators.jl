export NormalOp, normalOperator

struct NormalOp{S,D} 
  parent::S
  weights::D
end

function normalOperator(S, W=I)
  return NormalOp(S,W)
end

# Generic fallback -> TODO avoid allocations
function Base.:*(N::NormalOp, x::AbstractVector)

  #@info size(x)  size(N.parent)   size(N.weights)   

  d1 = N.parent*x
  d2 = N.weights*d1
  d3 = adjoint(N.parent)*d2
  return d3
end