export GradientOp

"""
    gradOp(T::Type, shape::NTuple{1,Int64})

1d gradient operator for an array of size `shape`
"""
GradientOp(T::Type, shape::NTuple{1,Int64}) = GradientOp(T,shape,1)

"""
    gradOp(T::Type, shape::NTuple{2,Int64})

2d gradient operator for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{2,Int64})
  return vcat( GradientOp(T,shape,1), GradientOp(T,shape,2) ) 
end

"""
    gradOp(T::Type, shape::NTuple{3,Int64})

3d gradient operator for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{3,Int64})
  return vcat( GradientOp(T,shape,1), GradientOp(T,shape,2), GradientOp(T,shape,3) ) 
end

"""
    gradOp(T::Type, shape::NTuple{N,Int64}, dim::Int64) where N

directional gradient operator along the dimension `dim`
for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{N,Int64}, dim::Int64) where N
  nrow = div( (shape[dim]-1)*prod(shape), shape[dim] )
  ncol = prod(shape)
  return LinearOperator{T}(nrow, ncol, false, false,
                          (res, x, α, β) -> (res .= α .* grad(x,shape,dim) ; res), 
                          (res, x, α, β) -> (res .= α .* grad_t(x,shape,dim) .+ (β .* res); res), 
                          (res, x, α, β) -> (res .= α .* grad_t(x,shape,dim) .+ (β .* res); res) )
end

# directional gradients
function grad(img::T, shape::NTuple{1,Int64}, dim::Int64) where T<:AbstractVector
  return img[1:end-1].-img[2:end]
end

function grad(img::T, shape::NTuple{2,Int64}, dim::Int64) where T<:AbstractVector
  img = reshape(img,shape)

  if dim==1
    return vec(img[1:end-1,:].-img[2:end,:])
  end
  
  return vec(img[:,1:end-1].-img[:,2:end])
end

function grad(img::T, shape::NTuple{3,Int64}, dim::Int64) where T<:AbstractVector
  img = reshape(img,shape)

  if dim==1
    return vec(img[1:end-1,:,:].-img[2:end,:,:])
  elseif dim==2
    return vec(img[:,:,1:end-1].-img[:,:,2:end])
  end
  
  return vec(img[:,:,1:end-1].-img[:,:,2:end])
end

# adjoint of directional gradients
function grad_t(g::T, shape::NTuple{1,Int64}, dim::Int64) where T<:AbstractVector
  nx = shape[1]
  img = similar(g,nx)
  img .= zero(eltype(g))

  img[1:nx-1] .= g
  img[2:nx] .-= g

  return img
end

function grad_t(g::T, shape::NTuple{2,Int64}, dim::Int64) where T<:AbstractVector
  nx,ny = shape
  img = similar(g,nx,ny)
  img .= zero(eltype(g))

  if dim==1
    g = reshape(g,nx-1,ny)
    img[1:nx-1,:] .= g
    img[2:nx,:] .-= g
  else
    g = reshape(g,nx,ny-1)
    img[:,1:ny-1] .= g
    img[:,2:ny] .-= g
  end

  return vec(img)
end

function grad_t(g::T, shape::NTuple{3,Int64}, dim::Int64) where T<:AbstractVector
  nx,ny,nz = shape
  img = similar(g,nx,ny,nz)
  img .= zero(eltype(g))

  if dim==1
    g = reshape(g,nx-1,nx,nz)
    img[1:nx-1,:,:] .= g
    img[2:nx,:,:] .-= g
  elseif dim==2
    g = reshape(g,nx,ny-1,nz)
    img[:,1:ny-1,:] .= g
    img[:,2:ny,:] .-= g
  else
    g = reshape(g,nx,nx,nz-1)
    img[:,:,1:nz-1] .= g
    img[:,:,2:nz] .-= g
  end
  return vec(img)
end