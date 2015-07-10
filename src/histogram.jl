# data and queries are represented as vectors in the histogram space

type Histogram <: Data
    weights::Vector
end

type HistogramQuery <: Query
    weights::Vector
end

type HistogramQueries <: Queries
    queries::Matrix
end

get(queries::HistogramQueries,i::QueryIndex) = HistogramQuery(queries.queries[:,i])
evaluate(query::HistogramQuery,h::Histogram) = dot(query.weights,h.weights)
evaluate(queries::HistogramQueries,h::Histogram) = queries.queries' * vec

function normalize!(h::Histogram)
    h.weights /= sum(h.weights)
    h
end

# if error is > 0, we want to increase weight on positive elements of queries[query]
# and decrease weight on negative elements. Magnitude of error determines step size.
function update!(q::HistogramQuery,h::Histogram,error::Float64)
    @simd for j = 1:length(h.weights)
        @inbounds h.weights[j] *= exp(error * q.weights[j] / 2.0)
    end
end

function histogram_initialize(queries::Queries,table::Tabular,parameters)
    d, n  = size(table.data)
    epsilon, iterations, repetitions, smart = parameters
    N = 2^d
    real = Histogram(table)
    if smart
        # spend half of epsilon on histogram initialization
        weights = Array(Float64,N)
        noise = rand(Laplace(0.0,1.0/(n*epsilon)),N)
        @simd for i = 1:N
             @inbounds weights[i] = max(real.weights[i]+noise[i]-1.0/(e*n*epsilon),0.0)
        end
        weights /= sum(weights)
        synthetic = Histogram(0.5 * weights + 0.5/N)
        epsilon = 0.5*epsilon
    else
        synthetic = Histogram(ones(N)/N)
    end
    real_answers = evaluate(queries,real)
    scale = 2*iterations/(epsilon*n)
    mwstate = MWState(real,synthetic,queries,real_answers,(Int=>Float)[],scale,repetitions)
    mwstate
end

function initialize(queries::HistogramQueries,table::Tabular,parameters)
    histogram_initialize(queries,table,parameters)
end

# convert 0/1 data matrix to its histogram representation
function Histogram(table::Tabular)
    d, n = size(table.data)
    histogram = zeros(2^d)
    for i = 1:n
        x = vec(table.data[:,i])
        # treat each row of binary values as one bit string
        bit_str = join(map(int, x), "")
        histogram[parseint(bit_str, 2) + 1] += 1.0
    end
    normalize!(Histogram(histogram))
end

function downcast_bits(x::Float32)
  bits(x)[1:3]  # only take first 3 bits
end

function HistogramFloat(table::Tabular)
    d, n = size(table.data)
    histogram = zeros(Uint8, 2^(3 * d))  ## ERROR not enough memory
    for i = 1:n
        x = vec(table.data[:,i])
        x = map(float32, x)  # WARNING casting to Float32
        x = map(downcast_bits, x)
        bit_str = join(x, "")
        idx = parseint(Uint128, bit_str, 2) + 1
        histogram[idx] += 1.0
    end
    normalize!(Histogram(histogram))
end

# convert histogram to 0/1 data matrix
function Tabular(histogram::Histogram,n::Int)
    N = length(histogram.weights)
    d = int(log(2,N))
    idx = wsample([0:N-1],histogram.weights,n)
    data_matrix = zeros(d,n)
    for i = 1:n
        data_matrix[:,i] = reverse(digits(idx[i],2,d))
    end
    Tabular(data_matrix)
end

function pad32bits(s::String)
  return join([s "00000000000000000000000000000"])
end

function bits2float(s::String)
  ## TODO this is a hack and is incorrect for negatives
  x = parseint(Int128, s, 2)
  return hex2num(hex(x))
end

function last3bits(s::String)
  return s[end-2:end]
end

function TabularFloat(histogram::Histogram,n::Int)
    N = length(histogram.weights)
    d = int(log(2,N) / 3)
    idx = wsample([0:N-1],histogram.weights,n)
    data_matrix = zeros(Float32, d,n)
    for i = 1:n
      coll = reverse(digits(idx[i],8,d))
      coll = map(bits, coll)
      # Only using 3 bits from each value
      coll = map(last3bits, coll)
      coll = map(pad32bits, coll)
      coll = map(bits2float,coll)
      data_matrix[:,i] = coll
    end
    Tabular(data_matrix)
end

