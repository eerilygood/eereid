import eereid as ee

g=ee.ghost(model=ee.models.PCB(
                    loss="softmax",
                    parts=4,
                    reduced_dim=256,
                    nonlinear='relu',
                    ),              #Use a simple convolutional model
           dataset=ee.datasets.pallet502(),         #Train on mnist dataset
           loss=ee.losses.triplet(),   #Use Extended Triplet loss (+d(p,n)) for training
           #prepros=[ee.prepros.resize((32,32)), ee.prepros.subsample(0.1)],   #To speed up training, use only 10% of samples
           triplet_count=10000,                   #To speed up training, use only 100 triplets for training
           crossval=False,
           epochs=100,
           patience=10,
           batch_size=20,
           step_size=20)                       #Enable crossvalidation



acc=g.evaluate()        #Evaluate the model

print(acc)
