# data and queries are represented as vectors in the histogram space

using Discretizers

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

chunk_size = 8

function bin_edge(coll::Vector)
  min = minimum(coll)
  max = maximum(coll)
  chunk = (max - min) / chunk_size
  edges = zeros(chunk_size)
  edges[1] = min
  edges[chunk_size] = max
  for i = 2:(chunk_size - 1)
    edges[i] = min + (chunk * (i - 1))
  end
  return edges
end

# bin_edges returns a dx8 matrix representing the 8-edge of each column
function bin_edges(t::Tabular)
  d, n = size(t.data)
  edges = zeros(d, chunk_size)
  for i = 1:d
    edges[i,:] = bin_edge(vec(t.data[i,:]))
  end
  return edges
end

function histogram_initialize(queries::Queries,table::Tabular,parameters)
    d, n  = size(table.data)
    epsilon, iterations, repetitions, smart = parameters
    N = 2^(d * 3)
    edges = bin_edges(table)
    real = HistogramFloat(table, edges)
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
    mwstate = MWState(real,synthetic,queries,real_answers,(Int=>Float)[],scale,repetitions,edges)
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

function HistogramFloat(table::Tabular, bin_edges)
    col, row = size(table.data)
    histogram = zeros(Uint8, 2^(3 * col))
    for i = 1:row
      encoded = zeros(col)
      for j = 1:col
        lindisc = LinearDiscretizer(vec(bin_edges[j,:]))
        encoded[j] = encode(lindisc, table.data[j,i])
      end

      encoded = map(int, encoded)
      encoded = map(bits, encoded)
      # take 3 bits only
      encoded = map(s -> s[end-2:end], encoded)

      bit_str = join(encoded, "")
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

function TabularFloat(histogram::Histogram, bin_edges, n::Int)
    N = length(histogram.weights)
    d = int(log(2,N) / 3)
    idx = wsample([0:N-1],histogram.weights,n)
    data_matrix = zeros(Float32, d,n)
    for i = 1:n
      coll = reverse(digits(idx[i],8,d))
      for j = 1:d
        lindisc = LinearDiscretizer(vec(bin_edges[j,:]))
        data_matrix[j,i] = decode(lindisc, coll[j])
      end
    end
    Tabular(data_matrix)
end

