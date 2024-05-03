import Preprocessing as pre
import model as m 
import config as c 

X = pre.X
y = pre.y
model = m.model 


epochs = c.epochs
batch_size = c.Batch_size

history = model.fit(X, y, epochs=epochs, batch_size=batch_size)



