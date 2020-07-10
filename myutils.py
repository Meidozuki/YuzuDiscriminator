
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output


def make_anime_dataset(img_path, batch_size, resize=None,labels=None,return_size=False,filter_fn=None):
    def parce_fn(path, *label):
        img= tf.io.read_file(path)
        img= tf.image.decode_image(img,channels=3,expand_animations=False)
        img= tf.image.resize(img, size, antialias=True)
        img= img/127.5 - 1
            
        return (img,)+label
    
    if resize is None:
        temp=Image.open(img_path[0])
        w,h=temp.size
        size=(h,w)
    else:
        size=resize
    
    if labels is None:
        dataset= tf.data.Dataset.from_tensor_slices(img_path)
    else:
        dataset= tf.data.Dataset.from_tensor_slices((img_path,labels))
    
    shuffle_buffer_size= max(batch_size*128, 2048)
    dataset= dataset.shuffle(shuffle_buffer_size)
    
    if filter_fn:
        dataset= dataset.filter(filter_fn)
    dataset= dataset.map(parce_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset= dataset.batch(batch_size, drop_remainder=True)
    dataset= dataset.repeat(-1)
    dataset= dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if (return_size):
        size=size+(3,)
        return dataset,size
    else:
        return dataset

def visualize_result(G,
                     G_input,
                     denorm_fn=None,
                     save=False,
                     save_path=None,
                     rewind=False,
                     show=True):
    '''
    help:
    用于可视化generator的输出。
    denorm_fn可输入函数或区间（默认为[-1,1]），过长tuple/list只取最前二者
    '''
    assert (not save) or save_path
    if (save == False): show=True
    
    image=G(G_input).numpy()
    block_size=int(np.floor(np.sqrt(min(100,image.shape[0]))))
    
    if denorm_fn is None:
        denorm_fn=[-1,1]
    if isinstance(denorm_fn,(tuple,list)):
        lower,upper=denorm_fn[0],denorm_fn[1]
        denorm_fn=lambda x:(x-lower)*(255.0/(upper-lower))
        
    image=denorm_fn(image).astype(np.uint8)
        
    final_image = np.array([])
    single_row = np.array([])
    for i in range(image.shape[0]):
        if single_row.size == 0:
            single_row = image[i, :, :, :]
        else:
            single_row = np.concatenate((single_row, image[i, :, :, :]), axis=1)
 
        if (i+1) % block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
 
            # reset single row
            single_row = np.array([])

    im=Image.fromarray(final_image)
    if save:
        im.save(save_path)
    if show:
        if rewind: clear_output()
        plt.imshow(im)
        plt.show()

def show_pred(x,
              y=None,
              denorm_fn=None,
              predict=False,
              D=None,
              categorical=False,
              adversal=False,
              figure_size=(10,12),
              single_thresh=2,
              min_num_per_line=4,
              max_num_shown=20):
    '''
    help:
    用于可视化discriminator的输出。
    denorm_fn可输入函数或区间（默认为[-1,1]），过长tuple/list只取最前二者
    categorical为True时，接收y为one-hot编码、输出argmax结果，为False时直接输出
    adversal表示是否为输出[label,validity]的discriminator
    数量小于single_thresh时单行输出数量不因min_num_per_line限制而缩小
    '''
    assert (not predict) or (D is not None)
    if predict:pred=D(x)
    if adversal:categorical=True
        
    if denorm_fn is None:
        denorm_fn=[-1,1]
    if isinstance(denorm_fn,(tuple,list)):
        lower,upper=denorm_fn[0],denorm_fn[1]
        denorm_fn=lambda x:(x-lower)*(255.0/(upper-lower))
        
    x=denorm_fn(x)
    if (x.shape[-1] == 1):img=np.squeeze(img,-1)
    
    plt.gcf().set_size_inches(figure_size)
    n=int(np.ceil(np.sqrt(x.shape[0])))
    if x.shape[0] >= single_thresh:
        n=max(n,min_num_per_line)
        
    loop_range=min(x.shape[0],max_num_shown)
    for i in range(loop_range):
        ax=plt.subplot(n,n,i+1)
        img=np.array(x[i]).astype(np.uint8)
        if (len(x.shape) == 3):
            #黑白单通道
            ax.imshow(img,cmap='gray')
        else:
            ax.imshow(img)
        if predict:
            #预测部分
            if pred.shape[-1] != 1:
                if adversal:
                    title="pred=%d" % (int(tf.argmax(pred[0][i])))
                else:
                    title="pred=%d" % (int(tf.argmax(pred[i])))
            else:
                title="validity=%.5f" % (float(pred[i]))
            #标签部分
            if y is None:
                ax.set_title(title)
            else:
                if categorical:
                    label=int(tf.argmax(y[i]))
                else:
                    label=int(y[i])
                ax.set_title(("true=%d," % label)+title , fontsize=13-n//2)
        else:
            if y is not None:
                ax.set_title("label=%d" % (y[i]))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def show_history(history,validation=False):
    ax=plt.subplot(1,1,1)
    ax.plot(history.history['loss'],linewidth=1)
    if validation:
        ax.plot(history.history['val_loss'],linewidth=1)
    ax.legend(['train_loss','val_loss'],loc='lower center')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax1=ax.twinx()
    ax1.plot(history.history['accuracy'],linewidth=2)
    if validation:
        ax1.plot(history.history['val_accuracy'],linewidth=2)
    ax1.legend(['train_acc','val_acc'],loc='upper center')
    plt.ylabel('acc')
    plt.show()