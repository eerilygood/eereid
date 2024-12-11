import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from eereid.tools import datasplit, build_triplets, build_Nlets, rankN, crossvalidation, add_tags, various_tags

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from eereid.modifier.mods import mods

from sklearn.metrics import roc_auc_score

import eereid as ee

from tqdm import tqdm

class ghost():
    def __init__(self,*tags,dataset=None, distance=None, loss=None, model=None, novelty=None, experiments=None,modifier=None,preproc=None,prepros=None,preprocessing=None, **kwargs):
        #add kwargs 
        self.dataset=ee.datasets.mnist()
        self.distance=ee.distances.euclidean()
        self.experiments={}
        self.loss=ee.losses.triplet()
        self.model=ee.models.conv()
        self.modifier=mods()
        self.novelty=None
        self.prepro={}

        self.logs=None

        for tag in tags:
            self.add(tag)

        if dataset is not None:
            self.set_dataset(dataset)
        if distance is not None:
            self.set_distance(distance)
        if loss is not None:
            self.set_loss(loss)
        if model is not None:
            self.set_model(model)
        if novelty is not None:
            self.set_novelty(novelty)
        if experiments is not None:
            if type(experiments) is list:
                for experiment in experiments:
                    self.add_experiment(experiment)
            else:
                self.add_experiment(experiments)
        if modifier is not None:
            if type(modifier) is list:
                for mod in modifier:
                    self.add_modifier(mod)
            else:
                self.add_modifier(modifier)
        if prepros is not None:
            if type(prepros) is list:
                for prepro in prepros:
                    self.add_prepro(prepro)
            else:
                self.add_prepro(prepros)
        if preproc is not None:
            if type(preproc) is list:
                for prepro in preproc:
                    self.add_prepro(prepro)
            else:
                self.add_prepro(preproc)
        if preprocessing is not None:
            if type(preprocessing) is list:
                for prepro in preprocessing:
                    self.add_prepro(prepro)
            else:
                self.add_prepro(preprocessing)

        self.add(kwargs)

    def _log(self,msg,importance=1):
        self.mods().log(msg,importance)

    def explain(self):
        data=[["Ghost ReID experiment",0],
              ["Dataset:",1],
              [self.dataset.explain(),2],
              ["Model:",1],
              [self.model.explain(),2],
              ["Loss:",1],
              [self.loss.explain(),2],
              ["Distance:",1],
              [self.distance.explain(),2],
              ["Preprocessings:",1]]
        for prepro in self.prepro.values():
            data.append([prepro.explain(),2])
        if not self.novelty is None:
            data.append(["Novelty Detection:",1])
            data.append([self.novelty.explain(),2])
        data.append(["Modifiers:",1])
        data.append([self.mods().explain(),2])


        return various_tags(data)



    def add(self,tag):
        if type(tag) is dict:
            for key,value in tag.items():
                self.modifier[key]=value
            return self
        species=tag.species()
        if species=='dataset':
            self.set_dataset(tag)
        elif species=='distance':
            self.set_distance(tag)
        elif species=='experiment':
            self.add_experiment(tag)
        elif species=='loss':
            self.set_loss(tag)
        elif species=='model':
            self.set_model(tag)
        elif species=='modifier':
            self.add_modifier(tag)
        elif species=='novelty':
            self.set_novelty(tag)
        elif species=='prepro':
            self.add_prepro(tag)
        else:
            raise ValueError("Unknown species encountered",species,tag)
        return self

    def set_dataset(self,dataset):
        self.dataset=dataset
        return self

    def set_distance(self,distance):
        self.distance=distance
        return self

    def set_loss(self,loss):
        self.loss=loss
        return self

    def set_model(self,model):
        self.model=model
        return self

    def set_novelty(self,novelty):
        self.novelty=novelty
        return self

    def add_experiment(self,experiment):
        self.experiments[experiment.ident()]=experiment
        return self

    def add_modifier(self,modifier):
        self.modifier.add(modifier)
        return self

    def add_prepro(self,prepro):
        self.prepro[prepro.ident()]=prepro
        return self


    def __setitem__(self,key,value):
        self.add({key:value})
        return self

    def __call__(self,*tags):
        for tag in tags:
            self.add(tag)
        return self

    def mods(self):
        return self.modifier

    def _preprocess(self):
        self._log("Starting preprocessing",1)
        tasks=[prepro for prepro in self.prepro.values() if prepro.stage()=="general"]
        tasks.sort(key=lambda x:x.order())
        for task in tasks:
            self._log(f"Applying preprocessing {task.ident()}",0)
            self.x,self.y=task.apply(self.x,self.y,self)
    def apply_preprocessing(self,data, labels=None):
        self._log("Starting preprocessing for the new data",0)
        x,y=data,labels
        if y is None:
            y=np.zeros(len(x))
        tasks=[prepro for prepro in self.prepro.values() if prepro.stage()=="general"]
        tasks.sort(key=lambda k:k.order())
        for prepro in tasks:
            self._log(f"Applying preprocessing {prepro.ident()}",0)
            x,y=prepro.apply(x,y,None)
        if labels is None:
            return x
        return x,y
    def apply_later_preprocessing(self,data, labels=None):
        self._log("Starting preprocessing for the new data",0)
        x,y=data,labels
        if y is None:
            y=np.zeros(len(x))
        tasks=[prepro for prepro in self.prepro.values() if prepro.stage()=="general" and prepro.apply_always()]
        tasks.sort(key=lambda k:k.order())
        for prepro in tasks:
            self._log(f"Applying preprocessing {prepro.ident()}",0)
            x,y=prepro.apply(x,y,None)
        if labels is None:
            return x
        return x,y
    def _preprocess_train(self):
        self._log("Starting training data preprocessing",1)
        tasks=[prepro for prepro in self.prepro.values() if prepro.stage()=="train"]
        tasks.sort(key=lambda x:x.order())
        for task in tasks:
            self._log(f"Applying preprocessing {task.ident()}",0)
            self.tx,self.ty=task.apply(self.tx,self.ty,self)

    def _basic_data_loading(self):
        self._log("Starting basic data loading",1)
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._log(f"Got input shape {self.input_shape}",1)
        self._preprocess()
        if self.novelty is None:
            self._log("Splitting data into training, query and gallery sets",1)
            self.tx,self.ty,self.qx,self.qy,self.gx,self.gy=datasplit(self.x,self.y,self.mods(),novelty=False)
        else:
            self._log("Splitting data into training, query, gallery and novelty sets",1)
            self.tx,self.ty,self.qx,self.qy,self.gx,self.gy,self.nx=datasplit(self.x,self.y,self.mods(),novelty=True)
        self._preprocess_train()
    def _direct_data_loading(self):
        self._log("Starting data loading without splits",1)
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._log(f"Got input shape {self.input_shape}",1)
        self._preprocess()
        self.tx,self.ty=self.x,self.y
        self._preprocess_train()
    def _crossval_data_loading(self):
        self._log("Starting crossvalidation data loading",1)
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._log(f"Got input shape {self.input_shape}",1)
        self._preprocess()
        fold=0
        if self.novelty is None:
            self._log("Splitting data into training, query and gallery sets",1)
            for self.tx,self.ty,self.qx,self.qy,self.gx,self.gy in crossvalidation(self.x,self.y,self.mods(),novelty=False):
                self._log(f"Starting fold {fold}",1)
                self._preprocess_train()
                yield fold
                fold+=1
        else:
            self._log("Splitting data into training, query, gallery and novelty sets",1)
            for self.tx,self.ty,self.qx,self.qy,self.gx,self.gy,self.nx in crossvalidation(self.x,self.y,self.mods(),novelty=True):
                self._log(f"Starting fold {fold}",1)
                self._preprocess_train()
                yield fold
                fold+=1

    def _crossval_data_loading_from_file(self):
        #under construction. Not sure if needed
        self._log("Starting crossvalidation data loading from files",1)
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._log(f"Got input shape {self.input_shape}",1)
        self._preprocess()
        fold=0
        if self.novelty is None:
            self._log("Splitting data into training, query and gallery sets",1)
            for self.tx,self.ty,self.qx,self.qy,self.gx,self.gy in crossvalidation(self.x,self.y,self.mods(),novelty=False):
                self._log(f"Starting fold {fold}",1)
                self._preprocess_train()
                yield fold
                fold+=1
        else:
            self._log("Splitting data into training, query, gallery and novelty sets",1)
            for self.tx,self.ty,self.qx,self.qy,self.gx,self.gy,self.nx in crossvalidation(self.x,self.y,self.mods(),novelty=True):
                self._log(f"Starting fold {fold}",1)
                self._preprocess_train()
                yield fold
                fold+=1
        
    def _create_model(self):
        self._log("Building the model",1)
        self.model.build(self.input_shape,self.loss.siamese_count(),self.mods())
        stream=io.StringIO()
        self.model.summary(print_fn=lambda x:stream.write(x+'\n'))
        self._log("Model summary:\n"+stream.getvalue(),0)


    def _pretrain_prediction(self):
        pretrain_epochs=self.mods()("pretrain_epochs",1)
        pretrain_loss=self.mods()("pretrain_los","categorical_crossentropy")
        pretrain_optimizer=self.mods()("pretrain_optimizer","adam")

        self._log(f"Pretraining the model for {pretrain_epochs} epochs",1)

        inp=keras.Input(shape=self.input_shape)
        classes=list(set(self.ty))
        classes.sort()
        classes=np.array(classes)
        onehot=np.eye(len(classes))
        class_to_vec={cls:vec for cls,vec in zip(classes,onehot)}
        onehot=np.array([class_to_vec[cls] for cls in self.ty])
        q=self.model.submodel(inp)
        q=keras.layers.Dense(len(classes),activation='softmax')(q)
        model=keras.Model(inp,q)

        model.compile(loss=pretrain_loss,optimizer=pretrain_optimizer)
        model.summary()
        model.fit(self.tx,onehot,epochs=pretrain_epochs)


    def _train_model(self):
        optimizer=self.mods()("optimizer","adam")
        epochs=self.mods()("epochs",10)
        batch_size=self.mods()("batch_size",32)
        validation_split=self.mods()("validation_split",0.1)
        early_stopping=self.mods()("early_stopping",True)
        patience=self.mods()("patience",5)
        restore_best_weights=self.mods()("restore_best_weights",True)
        terminate_on_nan=self.mods()("terminate_on_nan",True)
        callbacks=list(self.mods()("callbacks",[]))
        kwargs=dict(self.mods()("fit_kwargs",{}))
        verbose=self.mods()("verbose","auto")
        learning_rate=self.mods()("learning_rate",0.001)


        loss=self.loss.build(self.mods())
        self._log("Compiling the model",1)
        self.model.compile(loss=loss,optimizer=optimizer)

        #deprecated
        #K.set_value(self.model.model.optimizer.learning_rate,learning_rate)
        #replacement
        self.model.model.optimizer.learning_rate.assign(learning_rate)

        self._log("Building the training data",1)
        Nlets, labels=build_Nlets(self.tx,self.ty,self.loss.Nlet_string(),self.mods())

        #print(Nlets.shape,labels.shape)
        #print(self.model.model.input_shape, self.model.model.output_shape)
        #print(self.model.submodel.input_shape, self.model.submodel.output_shape)

        #exit()

        if early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(patience=patience,restore_best_weights=restore_best_weights))
        if terminate_on_nan:
            callbacks.append(keras.callbacks.TerminateOnNaN())

        self._log("Training the model",1)
        self.logs=self.model.fit(Nlets,labels,epochs=epochs,batch_size=batch_size,validation_split=validation_split,callbacks=callbacks,verbose=verbose,**kwargs)
        self._log("Training complete",1)
        if self.logs:
            self._log("Training logs:",0)
            self._log(str(self.logs.history),0)
        return self.logs

    def _embed(self,data):
        #embedding but no preprocess
        return self.model.embed(data)

    def _create_embeddings(self):
        self._log("Creating embeddings using the trained model",1)
        self.emb =self._embed(self.tx)
        self.qemb=self._embed(self.qx)
        self.gemb=self._embed(self.gx)
        if self.novelty is not None:
            self.nemb=self._embed(self.nx)

    def _all_available_data(self):
        ret={}
        poss=["x","y","tx","ty","qx","qy","gx","gy","nx","emb","qemb","gemb","nemb","input_shape"]
        for pos in poss:
            if hasattr(self,pos):
                ret[pos]=getattr(self,pos)
        return ret

    def load_model(self,pth=None):
        if pth is None:
            pth=self.mods()("model_file","eereid_model")
        self._log(f"Loading model from {pth}",1)
        self.set_model(ee.models.load_model(pth))
        self._log("Model loaded",0)

    def save_model(self,pth=None):
        if pth is None:
            pth=self.mods()("model_file","eereid_model")
        self._log(f"Saving model to {pth}",1)
        self.model.save_model(pth)
        self._log("Model saved",0)

    def save_data(self,pth):
        self._log(f"Saving data to {pth}",1)
        np.savez_compressed(pth,**self._all_available_data())
        self._log("Data saved",0)

    def load_data(self,pth):
        self._log(f"Loading data from {pth}",1)
        self.set_dataset(ee.datasets.load_data(pth))
        self._log("Data loaded",0)

    def _basic_accuracy(self):
        self._log("Starting the evaluation",1)
        self._create_embeddings()

        distance=self.distance

        self._log("Calculating the accuracy",1)
        acc=rankN(self.qemb,self.qy,self.gemb,self.gy,distance=distance)
        self._log(f"Accuracy calculated:",0)
        self._log(str(acc),0)

        if self.novelty is not None:
            self._log("Calculating the novelty detection accuracy",1)
            self._log("Creating the novelty model",1)
            self._log("Training the novelty model",1)
            self.novelty.inherit_info(self)
            self.novelty.create_model(self.gemb)
            normal=self.qemb
            abnormal=self.nemb
            test=np.concatenate([normal,abnormal])
            label=np.concatenate([np.zeros(len(normal)),np.ones(len(abnormal))])
            self._log("Evaluating the novelty model",1)
            acc["auc"]=roc_auc_score(label,self.novelty.predict(test))
            self._log(f"Novelty detection AUC calculated: {acc['auc']}",0)

        

        return acc

    def evaluate(self):
        self._log("Evaluating the following model\n"+self.explain(),1)
        self._basic_data_loading()
        if self.mods().hasattr("crossval"):
            return self._crossval_eval()
        elif self.mods().hasattr("repeated"):
            return self._repeated_eval(self.mods()("repeated"))
        else:
            return self._singular_eval()

    def _singular_eval(self):
        self._create_model()
        if self.mods().hasattr("pretrain"):self._pretrain_prediction()
        self._train_model()
        acc=self._basic_accuracy()

        return acc

    def train(self):
        self._log("Training the following model",1)
        self._log(self.explain(),1)
        self._direct_data_loading()
        self._create_model()
        if self.mods().hasattr("pretrain"):
            self._pretrain_prediction()
        self._train_model()
        if self.mods().hasattr("model_file"):
            self.save_model()

    def assert_trained(self):
        if self.mods().hasattr("model_file") and os.path.exists(self.mods()("model_file")):
            self._log("Model file found, loading the model",1)
            self.load_model()
        if self.model.trained==False:
            self._log("Model not trained, training the model",1)
            self.train()

    def embed(self,data):
        self._log("Embedding data",0)
        self.assert_trained()
        data=self.apply_later_preprocessing(data)
        return self.model.embed(data)

    def clear_gallery(self):
        self._log("Clearing the gallery",1)
        self.gemb=[]
        self.gx=[]
        self.gy=[]

    def add_to_gallery(self,data,labels):
        self._log(f"Adding {len(labels)} samples to the gallery",1)
        self.assert_trained()
        data,labels=self.apply_later_preprocessing(data,labels)
        self.gemb=np.concatenate([self.gemb,self.embed(data)],axis=0)
        self.gx=np.concatenate([self.gx,data],axis=0)
        self.gy=np.concatenate([self.gy,labels],axis=0)

    def predict(self,data):
        self._log("Predicting labels for the data",0)
        emb=self.embed(data)
        ret=[]
        for e in tqdm(emb):
            dist=self.distance.multi_distance(self.gemb,e)
            ret.append(self.gy[np.argmin(dist)])
        return np.array(ret)





    def _repeated_eval(self,n):
        self._log("Starting repeated evaluation",1)
        if type(n) is bool:
            n=10
        #should be done through modifier
        accs=None
        for i in range(n):
            self._log(f"Starting repetition {i}",1)
            acc=self._singular_eval()
            if accs is None:
                accs=acc
            else:
                accs=accs+acc
        return accs

    def _crossval_eval(self):
        self._log("Starting crossvalidation evaluation",1)
        #should be done through modifier
        accs=None
        for i in self._crossval_data_loading():
            self._log(f"Starting fold {i}",1)
            if self.mods().hasattr("repeated"):
                acc=self._repeated_eval(self.mods()("repeated"))
            else:
                acc=self._singular_eval()
            if accs is None:
                accs=acc
            else:
                accs=accs+acc
        return accs


    def plot_embeddings(self,alpha=1.0,max_classes=5,plot_linewidth=0.5):
        self._log("Plotting the embeddings",1)
        from sklearn.decomposition import PCA

        pca=PCA(n_components=2)
        emb=pca.fit_transform(self.emb)
        qemb=pca.transform(self.qemb)
        gemb=pca.transform(self.gemb)

        # FA = FactorAnalysis(n_components=2)
        # emb = FA.fit_transform(self.emb)
        # qemb = FA.transform(self.qemb)
        # gemb = FA.transform(self.gemb)

        unique_ty = list(set(self.ty))
        selected_ty = unique_ty[:max_classes]

        # mask = np.isin(self.ty, selected_classes)
        # mask_emb = emb[mask]
        # mask_ty = np.array(self.ty)[mask]


        classes=list(set(self.y))
        vmin,vmax=min(classes),max(classes)

        mn=np.min([np.min(emb,axis=0),np.min(qemb,axis=0),np.min(gemb,axis=0)],axis=0)
        mx=np.max([np.max(emb,axis=0),np.max(qemb,axis=0),np.max(gemb,axis=0)],axis=0)


        plt.subplot(1,2,1)

        plt.scatter(emb[:,0],emb[:,1],c=self.ty,alpha=alpha,vmin=vmin,vmax=vmax)
        # plt.plot(mask_emb[:,0],mask_emb[:,1],'-o')

        for i in range(len(selected_ty)):
            mask = np.isin(self.ty, selected_ty[i])
            mask_emb = emb[mask]
            plt.plot(mask_emb[:,0],mask_emb[:,1],'-o',linewidth=plot_linewidth)

        plt.xlim(mn[0],mx[0])
        plt.ylim(mn[1],mx[1])
        plt.xticks([])
        plt.yticks([])
        plt.title("Training")
        plt.xlabel("Principal Component")
        plt.ylabel("Principal Component")
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.scatter(qemb[:,0],qemb[:,1],c=self.qy,alpha=alpha,vmin=vmin,vmax=vmax)
        plt.scatter(gemb[:,0],gemb[:,1],c=self.gy,alpha=alpha,vmin=vmin,vmax=vmax)

        unique_qy = list(set(self.qy))
        selected_qy = unique_qy[:max_classes]

        unique_gy = list(set(self.gy))
        selected_gy = unique_gy[:max_classes]

        for i in range(len(selected_qy)):
            mask = np.isin(self.qy, selected_qy[i])
            mask_qemb = qemb[mask]
            # plt.plot(mask_qemb[:,0],mask_qemb[:,1],'-o')
            mask_gemb = gemb[np.isin(self.gy, selected_gy[i])]
            # plt.plot(mask_gemb[:,0],mask_gemb[:,1],'-o')
            mask_qgemb = np.concatenate([mask_qemb, mask_gemb])
            plt.plot(mask_qgemb[:,0],mask_qgemb[:,1],'-o')
        plt.xlim(mn[0],mx[0])
        plt.ylim(mn[1],mx[1])
        plt.xticks([])
        plt.yticks([])
        plt.title("Query + Gallery")
        plt.xlabel("Principal Component")

        plt.colorbar()

    def plot_loss(self,log=False):
        self._log("Plotting the loss",1)
        assert self.logs is not None, "To plot the loss of a model, please train it first"
        h=self.logs.history
        colors=["darkorchid","olivedrab","firebrick","royalblue","goldenrod","olivedrab","lightseagreen","darkorchid"]
        rename={"loss":"Training","val_loss":"Validation"}
        for i,(key,y) in enumerate(h.items()):
            x=np.arange(1,len(y)+1)
            if key in rename:
                key=rename[key]

            plt.plot(x,y,label=key,color=colors[i%len(colors)],linewidth=2)
            if len(y)<100:
                plt.plot(x,y,"o",color=colors[i%len(colors)],markersize=8)
            plt.axvline(np.argmin(y)+1,color=colors[i%len(colors)],linestyle="--",alpha=0.5)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if log:
            plt.yscale("log")
        plt.legend(frameon=True,framealpha=0.8)

    def plot_match(self, sample, true_label=None, n=10):
        
        if type(sample) is int:
            sample,sample0=self.qx[sample],sample
            if true_label is None:
                true_label=self.qy[sample0]
        
        embed=self.embed(np.array([sample]))
        dist=self.distance.multi_distance(self.gemb,embed)
        idx=np.argsort(dist)[:n]
        fig,ax=plt.subplots(1,n+1,figsize=(n*2,2))
        ax[0].imshow(sample.reshape(self.input_shape),cmap="gray")
        ax[0].title.set_text("ID: "+str(true_label))
        ax[0].axis("off")
        for i in range(n):
            ax[i+1].imshow(self.gx[idx[i]].reshape(self.input_shape),cmap="gray")
            ax[i+1].title.set_text("ID: "+str(self.gy[idx[i]]))
            ax[i+1].axis("off")
            if true_label is not None:
               # ax[i+1].set_title(f"True: {true_label}, Pred: {self.gy[idx[i]]}")
                if true_label==self.gy[idx[i]]:
                    color="limegreen"
                else:
                    color="red"
                axc=ax[i+1]
                pos = axc.get_position()
                rect = patches.Rectangle((pos.x0, pos.y0), pos.width, pos.height, linewidth=5, edgecolor=color, facecolor='none')
                fig.add_artist(rect)

    def plot_activation_heatmap(self,image,overlay=0.5):
        model=self.model.submodel
        if type(image) is int:
            image=self.qx[image]
        image=np.expand_dims(image,0)
        def find_last_conv_layer_name(model):
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower():
                #if hasattr(layer,"output_shape") and len(layer.output_shape) == 4:
                    return layer.name
            raise ValueError("Could not find the last convolutional layer.")
        # create a model that maps the input image to the activations
        # of the last convolutional layer
        activation_model = keras.models.Model(
            model.inputs, model.get_layer(find_last_conv_layer_name(model)).output
        )
        # activation_model.summary()
        # Get the feature maps for the input image
        conv_layer_output = activation_model(image)    
    
        # Square the feature maps and sum across channels
        # We want to sum across the channels (axis=-1)
        activation_map = tf.reduce_sum(tf.square(conv_layer_output), axis=-1)
    
        # Normalize the activation map between 0 and 1 for visualization
        activation_map = tf.maximum(activation_map, 0)  # Ensure non-negative values
        heatmap = activation_map[0] 
        heatmap = heatmap / tf.math.reduce_max(heatmap)  # Normalize      
        # Scale heatmap
        heatmap = tf.expand_dims(heatmap, axis=-1)  # Now it has shape [height, width, 1]
        heatmap = tf.image.resize(heatmap, (image.shape[1], image.shape[2]),method='bilinear')
        heatmap = tf.squeeze(heatmap, axis=-1)  # Back to shape [height, width]         

        htm=heatmap.numpy()
        image=image[0]
        image=image[::-1]
        while len(image.shape)>2:
            image=np.mean(image,axis=-1)

        plt.imshow(image,cmap="gray") 
        plt.imshow(htm,alpha=overlay,cmap="jet", extent=[0,len(image[0]),0,len(image)])



    def __add__(self,other):
        self._log("Adding two ghosts and building an haunting (ensemble)",1)
        if type(self) is haunting and type(other) is haunting:
            return haunting(*self.objs,*other.objs)
        if type(self) is haunting and type(other) is not haunting:
            return self.add_objs(other)
        if type(self) is not haunting and type(other) is haunting:
            return other.add_objs(self)
        return haunting(self,other)



class haunting(ghost):
    def __init__(self,*objs):
        self.objs=list(objs)
        super().__init__()
        self.dataset=self.search_through_objs("dataset")#objs[0].dataset
        self.distance=self.search_through_objs("distance")#objs[0].distance

    def add_objs(self,*objs):
        for obj in objs:
            self.objs.append(obj)
        return self

    def search_through_objs(self,attr):
        for obj in self.objs:
            if hasattr(obj,attr) and getattr(obj,attr) is not None:
                return getattr(obj,attr)
        return None

    def _log(self,msg,importance=1):
        for obj in self.objs:
            obj._log(msg,importance)
        super()._log(msg,importance)


    def _basic_data_loading(self):
        self._log("Starting ensemble type data loading",1)
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._preprocess()
        for obj in self.objs:
            obj.x=self.x
            obj.y=self.y
            obj.input_shape=self.input_shape
            obj._preprocess()
            obj.tx,obj.ty,obj.qx,obj.qy,obj.gx,obj.gy=datasplit(obj.x,obj.y,obj.mods())
            obj._preprocess_train()

        self.qx=self.objs[0].qx
        self.gx=self.objs[0].gx

    def _embed(self,data):
        self._log("Embedding data using the ensemble",0)
        return np.concatenate([obj.model.embed(data) for obj in self.objs],axis=1)

    def _singular_eval(self):
        self._log("Starting ensemble type evaluation",1)
        for obj in self.objs:
            obj._create_model()
            if obj.mods().hasattr("pretrain"):obj._pretrain_prediction()
            obj._train_model()
        self.qy=self.objs[0].qy
        self.gy=self.objs[0].gy
        acc=self._basic_accuracy()
        return acc

    def _crossval_data_loading(self):
        self._log("Starting ensemble type crossvalidation data loading",1)
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._preprocess()
        fold=0
        for self.tx,self.ty,self.qx,self.qy,self.gx,self.gy in crossvalidation(self.x,self.y,self.mods()):
            self._log(f"Starting fold {fold}",1)
            for obj in self.objs:
                obj.x=self.x
                obj.y=self.y
                obj.input_shape=self.input_shape
                obj.tx,obj.ty,obj.qx,obj.qy,obj.gx,obj.gy=self.tx,self.ty,self.qx,self.qy,self.gx,self.gy
                obj._preprocess_train()
            yield fold
            fold+=1

    def _create_embeddings(self):
        self._log("Creating embeddings using the ensemble",1)
        self.emb =np.concatenate([obj._embed(obj.tx) for obj in self.objs],axis=1)
        self.qemb=np.concatenate([obj._embed(obj.qx) for obj in self.objs],axis=1)
        self.gemb=np.concatenate([obj._embed(obj.gx) for obj in self.objs],axis=1)

    def __add__(self,other):
        self._log("Adding two ghosts and building an haunting (ensemble)",1)
        if type(self) is haunting and type(other) is haunting:
            return haunting(*self.objs,*other.objs)
        if type(self) is haunting and type(other) is not haunting:
            return self.add_objs(other)
        if type(self) is not haunting and type(other) is haunting:
            return other.add_objs(self)
        return haunting(self,other)

    def explain(self):
        ret=[[f"Haunting (Ensemble ghost) experiment, build from {len(self.objs)} ghosts.",0],
             ["Dataset:",1],
             [self.dataset.explain(),2],
             ["Distance:",1],
             [self.distance.explain(),2]]
        for i,obj in enumerate(self.objs):
            ret.append([f"Submodel {i+1}:",1])
            ret.append([obj.explain(),2])
        return various_tags(ret)
    
    def embed(self,data):
        self._log("Embedding data",0)
        # self.assert_trained()
        for obj in self.objs:
            obj.assert_trained()
        data=self.apply_later_preprocessing(data)

        return self._embed(data)
    
    def plot_activation_heatmaps(self,image):
        for obj in self.objs:
            obj.plot_activation_heatmap(image)
    
    def plot_lossess(self):
        for obj in self.objs:
            obj.plot_loss()








