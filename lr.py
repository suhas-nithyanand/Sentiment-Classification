import numpy as np
import theano
import theano.tensor as T
from sentiment_reader import SentimentCorpus
import six.moves.cPickle as pickle
from sklearn.metrics import f1_score
from timeit import default_timer as timer

start = timer() 

dataset = SentimentCorpus()

#debugging
print 'x',np.shape(dataset.train_X),'y', np.shape(dataset.train_y),'dict', len(dataset.feat_dict),'count', len(dataset.feat_counts)

n_classes = 2
n_instances = 1600
n_feats = len(dataset.feat_dict)
n_epoches = 3500

# input data for training 
train_x = np.matrix(dataset.train_X,dtype='float32')
train_y = np.asarray(dataset.train_y,dtype='int32')
train_y = np.reshape(train_y, 1600) 

print 'y shape',np.shape(train_y)
   
# declare Theano symbolic variables
x = T.matrix("x")
#y = T.ivector("y")
y = T.lvector("y")
w = theano.shared(np.random.randn(n_feats,n_classes), name="w")
b = theano.shared(np.zeros(n_classes), name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# construct Theano expression graph
p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
xent = -T.mean(T.log(p_y_given_x)[T.arange(n_instances), y])
cost = xent + 0.01 * (w ** 2).sum()       # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
y_pred = T.argmax(p_y_given_x, axis=1)
error = T.mean(T.neq(y_pred, y))

# compile
train = theano.function(inputs=[x,y],
          outputs=[error, cost],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

# train
for i in range(n_epoches):
    error, cost = train(train_x, train_y)
    print 'Current error: %.4f | Current cost: %.4f' % (error, cost)

with open('best_model.pkl', 'wb') as f:
	pickle.dump(train, f)


# load the saved model
classifier = pickle.load(open('best_model.pkl'))

# compile a predictor function
predict_model = theano.function(
inputs=[x],
outputs=y_pred)

# We can test it on some examples from test 
test_x, test_y = dataset.test_X, dataset.test_y
predicted_values = predict_model(test_x)


end = timer()
print '\nRunning Time :',(end - start) 

''' Calucalating total accuracy '''
count = 0
for i in range(len(predicted_values)):
	if predicted_values[i] == test_y[i]:
		count = count + 1

F1_score = f1_score(test_y, predicted_values, average='macro')  

print '\n Overall Accuracy =', float(count)/ len(predicted_values)

print '\n F1 score', F1_score
				

