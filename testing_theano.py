import theano
from theano import tensor as T

# run no both GPU and CPU
theano.config.floatX = 'float32'

# initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1*x1+w0

# compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execute
net_input(2., 1., .5)
