import cv2

print("Baca image dengan menggunakan OpenCV "+ cv2.__version__)
ipb = cv2.imread("data/ipb.png")
ipb = cv2.resize(ipb, (550,400))
cv2.imshow("Ipb logo",ipb)

five = cv2.imread("data/5.png")
print(five.shape)
print(five.size)
cv2.imshow("Five", five)

#konversi image 
ipb_gray = cv2.cvtColor(ipb, cv2.COLOR_BGR2GRAY)
cv2.imshow("Ipb logo gray",ipb_gray)
cv2.waitKey(0)

#mendapatkan nilai pixel dari image berdasarkan posisi
pixels= five[100,100]
print(pixels)
print(five[165,185])
