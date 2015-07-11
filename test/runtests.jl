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
table = Tabular([1.0 2.0 3.0 4.0 5.0])
edges5 = PrivateMultiplicativeWeights.bin_edges(table)
@test HistogramFloat(table, edges5).weights == [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.4]

# Multiple columns
table = Tabular([1.0 0.0; 0.0 1.0])
edges = PrivateMultiplicativeWeights.bin_edges(table)
diag_weights = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0]
@test HistogramFloat(table, edges).weights == diag_weights

# random numbers, so can't guarantee values
private = TabularFloat(Histogram([0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.4]), edges5, 100).data
@test maximum(private) <= 5.0
@test minimum(private) >= 1.0

@test PrivateMultiplicativeWeights.bin_edge([1.0, 2.0, 3.0, 4.0, 5.0]) == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# test our hadamard basis vectors agree with Hadamard module
for j = 0:10
    for i = j:10
        @test PrivateMultiplicativeWeights.hadamard_basis_vector(j,i) == Hadamard.hadamard(2^i)[:,j+1]
    end
end

