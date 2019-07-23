# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 2019 in ShangHai

@author: HuQiTong, ShanDong University
"""
import os
import progressbar
import numpy as np
from PIL import Image
from time import localtime,strftime
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

class ImageMaker():
    """
    This class is used to make image made by small images
    Advantages:Automatic,Fast,Free
    Disvantage:Fixed-size,Big-Memory
    """
    def __init__(self,img_path,img_dir,total_img_path,imscale,ans_imsize,precision):
        #1.img_path:The image path that we are willing to make.
        #2.img_dir:The image folder path that to make the uppering image.
        #3.total_img_path:The path that we are willing to store the result image.
        #4.imscale:The scale to narrow the image.
        #5.ans_imsize:The actual size of small images.
        #6.precision:The richness of small image that to make the uppering image.
        #7.imsize:The actual number of images that to make the result picture.
        #8.indices:The nearest image's index of each pixel.
        #9.img_tree:The KDTree to transform pixels to images.
        setattr(self,'img_path',img_path)
        setattr(self,'img_dir',img_dir)
        setattr(self,'total_img_path',total_img_path)
        setattr(self,'imscale',imscale)
        setattr(self,'ans_imsize',ans_imsize)
        setattr(self,'precision',precision)
        setattr(self,'imsize',())
        setattr(self,'indices',[])
        setattr(self,'img_tree',None)
        
    def split_imgs(self):#This function is to travere images by using 'split_img'.
        #To travere img_path
        img_path=getattr(self,'img_path')
        if(not os.path.exists(img_path)):#To make sure that the path is vaild.
            print('The image path is not vaild')
            exit(0)
        
        #to travere images in img_dir.
        img_dir=getattr(self,'img_dir')
        setattr(self,'img_dir','tmp')
        img_names=os.listdir(img_dir)
        if(not os.path.exists(img_dir)):#To make sure that the path is vaild.
            print('The image folder path is not vaild\n')
            exit(0)
        if(not len(img_names)):#To make sure there are images in this folder.
            print('There are no pictures in the folder\n')
            exit(0)
        bar=progressbar.ProgressBar()
        bar.start()
        for i in range(len(img_names)):
            self.split_img(os.path.join(img_dir,img_names[i]))
            bar.update(i/len(img_names)*100)
        bar.finish()
        
    def split_img(self,img_path):#To crop the center of the image.
        image=Image.open(img_path)
        imsize=list(image.size)
        ans_imsize=getattr(self,'ans_imsize')

        #If there are 4 dimensions in some pictures,delete them
        if(np.shape(image)[2]!=3):
            if os.path.exists(img_path):
                return False

        #Rotate the image if the height is lower than the width.
        if(imsize[0]>imsize[1]):
            image=image.transpose(Image.ROTATE_90)
            [imsize[1],imsize[0]]=imsize

        #To calculate the center,width,height of the new image.
        imcenter=np.array(imsize)/2
        imlength_half=(int)(np.min([imsize[0]*ans_imsize[1]/ans_imsize[0],imsize[1]])/2)
        imwidth_half=(int)(imlength_half*ans_imsize[0]/ans_imsize[1])

        #To calcualte the left,right,bottom,top dot of the image.
        left=imcenter[0]-imwidth_half
        right=imcenter[0]+imwidth_half
        bottom=imcenter[1]-imlength_half
        top=imcenter[1]+imlength_half

        new_image=image.crop((left,bottom,right,top))
        try:#if we have the right to store the image,we will store it.
            new_image.save(os.path.join('tmp',os.path.basename(img_path)))
        except:
            print('We don\'t have the right to store image\n')
            exit(0)
    
    def resize_imgs(self):#To travere the images by using 'resize_img'.
        #To travere img_path.
        img_path=getattr(self,'img_path')
        ans_imsize=getattr(self,'ans_imsize')
        self.resize_img(img_path,())
        
        #To travere images in img_dir.
        img_dir=getattr(self,'img_dir')
        img_names=os.listdir(img_dir)
        bar=progressbar.ProgressBar()
        bar.start()
        for i in range(len(img_names)):
            self.resize_img(os.path.join('tmp',img_names[i]),ans_imsize)            
            bar.update(i/len(img_names)*100)
        bar.finish()
            

    def resize_img(self,img_path,new_size):#Resize the image in order that the image is a square
        if(not new_size):
            return True
        image=Image.open(img_path)
        new_image=image.resize(new_size)
        
        #if we have the right to store the image,we will store it.
        try:#if we have the right to store the image,we will store it.
            new_image.save(os.path.join('tmp',os.path.basename(img_path)))
        except:
            print('We don\'t have the right to store image\n')
            exit(0)
        
    def load_img_dir(self):#To get the information of the images from each images.
        img_dir=getattr(self,'img_dir')
        precision=getattr(self,'precision')
        img_names=os.listdir(img_dir)
        mean_RGB=np.zeros(shape=(len(img_names),3))
        
        #To calculate the mean value of the RGB of the image.
        bar=progressbar.ProgressBar()
        bar.start()
        for i in range(len(img_names)):
            image=Image.open(os.path.join('tmp',img_names[i]))
            immat=np.asarray(image)
            mean_RGB[i,:]=np.mean(np.mean(immat,0),0)
            bar.update(i/len(img_names)*100)
        bar.finish()
        #To get the nearest image by kdtree and get the similar image by kmeans.
        img_tree=KDTree(mean_RGB)
        kmeans=KMeans(n_clusters=precision).fit(mean_RGB)

        setattr(self,'img_tree',img_tree)
        setattr(self,'img_labels',kmeans.labels_)

    def similar_img(self,pic_index):#To get the similar image in the same cluster.
        img_labels=getattr(self,'img_labels')
        pic_type=img_labels[pic_index]
        similar_indexs=np.where(np.array(img_labels)==pic_type) #Randomly chose one of them.
        return similar_indexs[0][np.random.randint(0,len(similar_indexs[0]),1)[0]]

    def load_img(self):#To get the information of the image and resize the image
        img_path=getattr(self,'img_path')
        img_tree=getattr(self,'img_tree')
        imscale=getattr(self,'imscale')
        ans_imsize=getattr(self,'ans_imsize')

        image=Image.open(img_path)
        imsize=image.size
        
        #To get each node's pixel.
        image=image.resize(((int)(imsize[0]/imscale),(int)(imsize[1]*ans_imsize[0]/(imscale*ans_imsize[1]))))
        immat=np.asarray(image)
        immat=np.reshape(immat,(-1,3))

        distances,indices=img_tree.query(immat,return_distance=True) #Get the nearest image's index.
        bar=progressbar.ProgressBar()
        bar.start()
        for i in range(len(indices)):
            indices[i]=self.similar_img(indices[i]) #To make image more beautiful.
            bar.update(i/len(indices)*100)
        bar.finish()
        setattr(self,'imsize',image.size)
        setattr(self,'indices',indices)

    def splice_img(self):#Splice small images into a new image
        imsize=getattr(self,'imsize')
        ans_imsize=getattr(self,'ans_imsize')
        indices=getattr(self,'indices')
        img_dir=getattr(self,'img_dir')
        total_img_path=getattr(self,'total_img_path')

        img_names=os.listdir(img_dir)
        total_image=Image.new('RGB',(imsize[0]*ans_imsize[0],imsize[1]*ans_imsize[1]),'white')
        bar=progressbar.ProgressBar()
        bar.start()
        for i in range(len(indices)):
            image=Image.open(os.path.join('tmp',img_names[indices[i,0]]))
            width=i%imsize[0]#Get the rows and cols of the small images
            height=i//imsize[0]
            cropimage=image.copy()
            total_image.paste(cropimage,(width*ans_imsize[0],height*ans_imsize[1]))
            bar.update(i/len(indices)*100)
        bar.finish()
        try:#if we have the right to store the image,we will store it.
            total_img_name=strftime("%b-%d-%Y-%H-%M-%S",localtime())
            total_image.save(os.path.join(total_img_path,total_img_name+'.png'))
        except:
            print('We don\'t have the right to store image\n')
            exit(0)

    def delete_imgs(self):
        img_dir=getattr(self,'img_dir')
        img_names=os.listdir(img_dir)
        try:
            for img_name in img_names:
                os.remove(os.path.join('tmp',img_name))
        except:
            print('We don\'t have the right the delete the image')
            exit(0)

    def run(self):
        if(not os.path.exists('tmp')):
            os.mkdir('tmp')
        print('------Spliting the Images------\n')
        self.split_imgs()
        print('------Resizeing the Images------\n')
        self.resize_imgs()
        print('------Getting thr Information from thr Folder------\n')
        self.load_img_dir()
        print('------Getting the Information of thr Image------\n')
        self.load_img()
        print('------Splicing the Images------\n')
        self.splice_img()
        print('------Deleting the tmp Images------\n')
        self.delete_imgs()
        print('------Successfully------\n')

if __name__=='__main__':
    img_path='C:/Users/25598/Desktop/ImageMaker/IMG_3527.JPG'
    img_dir='C:/Users/25598/Desktop/ImageMaker/Test'
    total_img_path='C:/Users/25598/Desktop/ImageMaker'
    imscale=8
    ans_imsize=(45,80)
    precision=10
    ImageMaker(img_path,img_dir,total_img_path,imscale,ans_imsize,precision).run()