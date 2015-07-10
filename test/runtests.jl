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
@test HistogramFloat(Tabular(zeros(1,10))).weights == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
@test HistogramFloat(Tabular(ones(1,10))).weights == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Multiple columns
diag_weights = vec([0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.5 zeros(1, 55)])
@test HistogramFloat(Tabular([1 0; 0 1])).weights == diag_weights

@test TabularFloat(Histogram([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),10).data == zeros(1,10)
# not ones because of lost of precision
@test TabularFloat(Histogram([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),10).data == fill(1.0842e-19, 1,10)
@test TabularFloat(HistogramFloat(Tabular(fill(1000,1,10))),10).data == fill(1.0842e-19, 1,10)

# test our hadamard basis vectors agree with Hadamard module
for j = 0:10
    for i = j:10
        @test PrivateMultiplicativeWeights.hadamard_basis_vector(j,i) == Hadamard.hadamard(2^i)[:,j+1]
    end
end

