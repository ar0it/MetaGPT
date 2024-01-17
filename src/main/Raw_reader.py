import imagej

print("Hello World")
ij = imagej.init('sc.fiji:fiji')
print(ij.getVersion())
image_url = '/home/aaron/PycharmProjects/MetaGPT/raw_data/Image_8.czi'
jimage = ij.io().open(image_url)

print(jimage.getProperties().keySet())