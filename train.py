# -*- coding: utf-8 -*-

from Import import *
import MyLib as myl

#%%

use_im_num = 300
im_size = [64,64]
#%%

# A unit
x_input = Input(shape=(3,im_size[0],im_size[1]),dtype='float32', name='x_input')
A = Convolution2D(3, 3, 3, border_mode='same',name='A')(x_input)

# R unit 
e_input = Input(shape=(1,6,im_size[0],im_size[1]),dtype='float32', name='e_input')
r_input = Input(shape=(3,im_size[0],im_size[1]),dtype='float32', name='r_input')
R = LSTMConv2D(nb_filter=3, nb_row=4, nb_col=4,
                       dim_ordering="th", input_shape=(5,6,im_size[0],im_size[1]),
                       border_mode="same", return_sequences=False,name='R')(e_input)

# A hat unit
Ahat = Convolution2D(3, 3, 3, border_mode='same',name='Ahat')(R)

# E unit : pixcel layer (l=0) 
Relu = Activation('relu',name='Relu')
e0 = Relu(Merge(mode=lambda x: x[0] - x[1],output_shape=(3,im_size[0],im_size[1]),name='X-Ahat')([x_input,Ahat]))
e1 = Relu(Merge(mode=lambda x: x[1] - x[0],output_shape=(3,im_size[0],im_size[1]),name='Ahat-X')([x_input,Ahat]))
E =  Merge(mode='concat', concat_axis=1,name='E')([e0,e1])
#%%

# Model 
predict_m = Model(input=[e_input],output=[Ahat])
predict_m.compile(optimizer='rmsprop', loss='mean_squared_error')

model = Model(input=[x_input,e_input],output=[E])
model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
# visualization
plot(model, to_file='model.png') 
def get_output_layer(model, layer_name,n):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name].get_output_at(n)
    return layer          
#%%

x_train = []          
in_E = np.zeros((1,6,im_size[0],im_size[1]))
labels = np.zeros((1,6,im_size[0],im_size[1]))

print '\n---lead dataset Image \n'
data_dir = './image/move_circle/'
im_dir,dir_num = myl.ListDir(data_dir)
print dir_num,im_dir

im_sum = 0
for name in im_dir:
    print data_dir+name
    im = Image.open(data_dir+name)
    im = im.resize((im_size[1],im_size[0]))        
    im = np.asarray(im.convert('RGB'))
    #im = ImageOps.grayscale(im)   
    im = np.asarray(im)  
    x_train.append(im)  
    if len(x_train) > use_im_num:
            break      

print '\n---Image Nomarization & Reduction'
x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3,im_size[0],im_size[1])) /255
#%%

end_flag = False
for epoch in range(1000):
    print epoch
    
    image_num = 0
    for im_num in range(len(x_train)):
        print epoch,im_num
        in_im = x_train[im_num].reshape((1,3,im_size[0],im_size[1]))
        
        in_E = in_E.reshape((1,1,6,im_size[0],im_size[1]))
        model.fit({'x_input':in_im, 'e_input':in_E}, labels,
                  nb_epoch=1, batch_size=1)
        in_E = model.predict({'x_input':in_im, 'e_input':in_E})
        tmp_E = in_E.reshape((1,1,6,im_size[0],im_size[1]))
        out_im = predict_m.predict({'e_input':tmp_E})
        print in_E.shape
        image_num += 1
        
        if image_num > use_im_num:
            break

        print "show result"
        predict_image = get_output_layer(model, 'Ahat',0)
        print out_im.shape
        show_im = (out_im[0].reshape(im_size[0],im_size[1],3))
        cv2.imshow("predict",show_im)
        key = cv2.waitKey(1)
        if key in [myl.KEY_ESC, ord('q')]:
            end_flag = True
        
    if end_flag:
        break
        
print "---Loop end---"
    