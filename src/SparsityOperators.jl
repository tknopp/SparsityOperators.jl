module SparsityOperators

import Base: length, iterate, \
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using LinearOperators
using SparseArrays
using Random

using FFTW
using Wavelets

const Trafo = Union{AbstractMatrix, AbstractLinearOperator, Nothing}
const FuncOrNothing = Union{Function, Nothing}


include("FFTOp.jl")
include("DCTOp.jl")
include("DSTOp.jl")
include("WaveletOp.jl")
include("SamplingOp.jl")
include("WeightingOp.jl")

export linearOperator, linearOperatorList

linearOperator(op::Nothing,shape) = nothing

"""
  returns a list of currently implemented `LinearOperator`s
"""
function linearOperatorList()
  return ["DCT-II", "DCT-IV", "FFT", "DST", "Wavelet"]
end

"""
    linearOperator(op::AbstractString, shape)

returns the `LinearOperator` with name `op`.

# valid names
* `"FFT"`
* `"DCT-II"`
* `"DCT-IV"`
* `"DST"`
* `"Wavelet"`
"""
function linearOperator(op::AbstractString, shape)
  shape_ = tuple(shape...)
  if op == "FFT"
    trafo = FFTOp(ComplexF32, shape_, false) #FFTOperator(shape)
  elseif op == "DCT-II"
    shape_ = tuple(shape[shape .!= 1]...)
    trafo = DCTOp(ComplexF32, shape_, 2)
  elseif op == "DCT-IV"
    shape_ = tuple(shape[shape .!= 1]...)
    trafo = DCTOp(ComplexF32, shape_, 4)
  elseif op == "DST"
    trafo = DSTOp(ComplexF32, shape_)
  elseif op == "Wavelet"
    trafo = WaveletOp(shape_)
  else
    error("Unknown transformation")
  end
  trafo
end


end # module
