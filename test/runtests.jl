using PrivateMultiplicativeWeights
using Hadamard
using Base.Test

# test histogram transform
@test Histogram(Tabular(int(zeros(1,10)))).weights == [1.0,0.0]
@test Histogram(Tabular(int(ones(1,10)))).weights == [0.0,1.0]
@test Histogram(Tabular([1 0; 0 1])).weights == [0.0,0.5,0.5,0]
@test Histogram(Tabular([1 0 1; 0 1 0])).weights == [0.0,1.0,2.0,0]/3.0
@test Histogram(Tabular([1 0 1 1 ; 0 1 0 1])).weights == [0.0,1.0,2.0,1.0]/4.0
@test Histogram(Tabular([1 0 1 0 1; 0 1 0 0 1])).weights == [1.0,1.0,2.0,1.0]/5.0
@test Tabular(Histogram([1.0,0.0,0.0,0.0]),10).data == zeros(2,10)
@test Tabular(Histogram([0.0,0.0,0.0,1.0]),10).data == ones(2,10)

# One column
@test HistogramFloat16(Tabular(zeros(1,10))).weights == vec([1.0 zeros(1, 2^16 - 1)])
x = zeros(2^16)
x[parseint(bits(float16(1.0)), 2) + 1] = 1.0
@test HistogramFloat16(Tabular(ones(1,10))).weights == x

# Multiple columns
# TODO failed because out of memory
#=@test HistogramFloat16(Tabular([1 0; 0 1])).weights == [0.0,0.5,0.5,0]=#

# test our hadamard basis vectors agree with Hadamard module
for j = 0:10
    for i = j:10
        @test PrivateMultiplicativeWeights.hadamard_basis_vector(j,i) == Hadamard.hadamard(2^i)[:,j+1]
    end
end

