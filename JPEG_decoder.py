import sys
import numpy as np  # import NumPy (library for scientific computing, for array operations and mathematical functions.)
import os  # Importing an os library provides many functions that interact with the operating system, such as file and directory processing.
from PIL import Image  # Import Image module from PIL library for opening, processing and saving images.

class KJPEG:  # Define the KJPEG class, which seems to be a tool class for processing JPEG images.
    def __init__(self):  # construct functions, used to initialize an instance of the KJPEG class
        
        self.__dctA = np.zeros(shape=(8, 8))  # Initialize an 8x8 zero matrix inside the class for coefficient matrix of discrete cosine transform
        for i in range(8):  # Traverse each row of the matrix to calculate the value of DCT matrix
            c = 0
            if i == 0:  # Conditional judgment, a special calculation method for distinguishing the first row of DCT matrix.
                c = np.sqrt(1 / 8)  # Calculate the square root, part of the calculation of DCT matrix
            else:
                c = np.sqrt(2 / 8)  # Calculate the square root, part of the calculation of DCT matrix
            for j in range(8):  # loop is used to traverse each column of the matrix
                self.__dctA[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * 8))  # Initialize an 8x8 zero matrix inside the class for the coefficient matrix of discrete cosine transform.
        # Brightness quantization matrix
        self.__lq = np.array([
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ])
        # Chrominance quantization matrix
        self.__cq = np.array([
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ])
        # Label matrix type, lt is brightness matrix, ct is chroma matrix.
        self.__lt = 0
        self.__ct = 1
        # https://my.oschina.net/tigerBin/blog/1083549
        # Zig coding table
        self.__zig = np.array([
            0, 1, 8, 16, 9, 2, 3, 10,
            17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ])
        # Zag coding table
        self.__zag = np.array([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 41, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ])
    
    def read_jpeg(self, filepath):
        """Reads a JPEG file from the specified filepath."""
        self.image = Image.open(filepath)
        self.image_data = np.array(self.image)

    def save_to_bmp(self, filepath):
        """Saves the current image data to a BMP file at the specified filepath."""
        if self.image_data is not None:
            Image.fromarray(self.image_data).save(filepath, format='BMP')


    def __Rgb2Yuv(self, r, g, b):
        # Obtaining YUV matrix from image
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        return y, u, v

    def __Fill(self, matrix):
        # The length and width of the picture must be a multiple of 16 (the sampling length and width will be reduced by 1/2 and the block length and width will be reduced by 1/8).
        # Three sampling methods of image compression are 4:4:4, 4:2:2 and 4:2:0.
        fh, fw = 0, 0
        if self.height % 16 != 0:
            fh = 16 - self.height % 16
        if self.width % 16 != 0:
            fw = 16 - self.width % 16
        res = np.pad(matrix, ((0, fh), (0, fw)), 'constant',
                             constant_values=(0, 0))
        return res

    def __Encode(self, matrix, tag):
        # Fill the matrix first
        matrix = self.__Fill(matrix)
        # Cut the image matrix into 8*8 small blocks
        height, width = matrix.shape
        # Reduce for loop statements, and improve the efficiency of the algorithm by using numpy's own functions
        # Referring to Andrew Ng's open class video, numpy's function comes with parallel processing instead of serial processing like for loop
        shape = (height // 8, width // 8, 8, 8)
        strides = matrix.itemsize * np.array([width * 8, 8, width, 1])
        blocks = np.lib.stride_tricks.as_strided(matrix, shape=shape, strides=strides)
        res = []
        for i in range(height // 8):
            for j in range(width // 8):
                res.append(self.__Quantize(self.__Dct(blocks[i, j]).reshape(64), tag))
        return res

    def __Dct(self, block):
        # DCT transform
        res = np.dot(self.__dctA, block)  # Initialize an 8x8 zero matrix inside the class for the coefficient matrix of discrete cosine transform.
        res = np.dot(res, np.transpose(self.__dctA))  # Initialize an 8x8 zero matrix inside the class for the coefficient matrix of discrete cosine transform.
        return res

    def __Quantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res = np.round(res / self.__lq)
        elif tag == self.__ct:
            res = np.round(res / self.__cq)
        return res

    def __Zig(self, blocks):
        ty = np.array(blocks)
        tz = np.zeros(ty.shape)
        for i in range(len(self.__zig)):
            tz[:, i] = ty[:, self.__zig[i]]
        tz = tz.reshape(tz.shape[0] * tz.shape[1])
        return tz.tolist()

    def __Rle(self, blist):
        res = []
        cnt = 0
        for i in range(len(blist)):
            if blist[i] != 0:
                res.append(cnt)
                res.append(int(blist[i]))
                cnt = 0
            elif cnt == 15:
                res.append(cnt)
                res.append(int(blist[i]))
                cnt = 0
            else:
                cnt += 1
        # The situation of all zeros at the end
        if cnt != 0:
            res.append(cnt - 1)
            res.append(0)
        return res

    def Compress(self, filename):
        # Read the picture according to the path image_path and store it as RGB matrix.
        image = Image.open(filename)
        # Get picture width width and height
        self.width, self.height = image.size
        image = image.convert('RGB')
        image = np.asarray(image)
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        # Convert RGB image to YUV
        y, u, v = self.__Rgb2Yuv(r, g, b)
        # Encoding the image matrix
        y_blocks = self.__Encode(y, self.__lt)
        u_blocks = self.__Encode(u, self.__ct)
        v_blocks = self.__Encode(v, self.__ct)
        # Zig coding and RLE coding are carried out on small image blocks
        y_code = self.__Rle(self.__Zig(y_blocks))
        u_code = self.__Rle(self.__Zig(u_blocks))
        v_code = self.__Rle(self.__Zig(v_blocks))
        # Calculate VLI variable word length integer coding and write it into a file, and Huffman part is not implemented
        # Detailed explanation of principle：https://www.cnblogs.com/Arvin-JIN/p/9133745.html
        buff = 0
        tfile = os.path.splitext(filename)[0] + ".gpj"
        if os.path.exists(tfile):
            os.remove(tfile)
        with open(tfile, 'wb') as o:
            o.write(self.height.to_bytes(2, byteorder='big'))
            o.flush()
            o.write(self.width.to_bytes(2, byteorder='big'))
            o.flush()
            o.write((len(y_code)).to_bytes(4, byteorder='big'))
            o.flush()
            o.write((len(u_code)).to_bytes(4, byteorder='big'))
            o.flush()
            o.write((len(v_code)).to_bytes(4, byteorder='big'))
            o.flush()
        self.__Write2File(tfile, y_code, u_code, v_code)

    # https://blog.csdn.net/weixin_43690347/article/details/84146979
    def __Write2File(self, filename, y_code, u_code, v_code):
        with open(filename, "ab+") as o:
            buff = 0
            bcnt = 0
            data = y_code + u_code + v_code
            for i in range(len(data)):
                if i % 2 == 0:
                    td = data[i]
                    for ti in range(4):
                        buff = (buff << 1) | ((td & 0x08) >> 3)
                        td <<= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
                else:
                    td = data[i]
                    vtl, vts = self.__VLI(td)
                    for ti in range(4):
                        buff = (buff << 1) | ((vtl & 0x08) >> 3)
                        vtl <<= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
                    for ts in vts:
                        buff <<= 1
                        if ts == '1':
                            buff |= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
            if bcnt != 0:
                buff <<= (8 - bcnt)
                o.write(buff.to_bytes(1, byteorder='big'))
                o.flush()
                buff = 0
                bcnt = 0

    def __IDct(self, block):
        # IDCT transformation
        res = np.dot(np.transpose(self.__dctA), block)  # Initialize an 8x8 zero matrix inside the class for the coefficient matrix of discrete cosine transform
        res = np.dot(res, self.__dctA)  # Initialize an 8x8 zero matrix inside the class for the coefficient matrix of discrete cosine transform
        return res

    def __IQuantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res *= self.__lq
        elif tag == self.__ct:
            res *= self.__cq
        return res

    def __IFill(self, matrix):
        matrix = matrix[:self.height, :self.width]
        return matrix

    def __Decode(self, blocks, tag):
        tlist = []
        for b in blocks:
            b = np.array(b)
            tlist.append(self.__IDct(self.__IQuantize(b, tag).reshape(8 ,8)))
        height_fill, width_fill = self.height, self.width
        if height_fill % 16 != 0:
            height_fill += 16 - height_fill % 16
        if width_fill % 16 != 0:
            width_fill += 16 - width_fill % 16
        rlist = []
        for hi in range(height_fill // 8):
            start = hi * width_fill // 8
            rlist.append(np.hstack(tuple(tlist[start: start + (width_fill // 8)])))
        matrix = np.vstack(tuple(rlist))
        res = self.__IFill(matrix)
        return res

    def __ReadFile(self, filename):
        with open(filename, "rb") as o:
            tb = o.read(2)
            self.height = int.from_bytes(tb, byteorder='big')
            tb = o.read(2)
            self.width = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            ylen = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            ulen = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            vlen = int.from_bytes(tb, byteorder='big')
            buff = 0
            bcnt = 0
            rlist = []
            itag = 0
            icnt = 0
            vtl, tb, tvtl = None, None, None
            while len(rlist) < ylen + ulen + vlen:
                if bcnt == 0:
                    tb = o.read(1)
                    if not tb:
                        break
                    tb = int.from_bytes(tb, byteorder='big')
                    bcnt = 8
                if itag == 0:
                    buff = (buff << 1) | ((tb & 0x80) >> 7)
                    tb <<= 1
                    bcnt -= 1
                    icnt += 1
                    if icnt == 4:
                        rlist.append(buff & 0x0F)
                    elif icnt == 8:
                        vtl = buff & 0x0F
                        tvtl = vtl
                        itag = 1
                        buff = 0
                else:
                    buff = (buff << 1) | ((tb & 0x80) >> 7)
                    tb <<= 1
                    bcnt -= 1
                    tvtl -= 1
                    if tvtl == 0 or tvtl == -1:
                        rlist.append(self.__IVLI(vtl, bin(buff)[2:].rjust(vtl, '0')))
                        itag = 0
                        icnt = 0
        y_dcode = rlist[:ylen]
        u_dcode = rlist[ylen:ylen+ulen]
        v_dcode = rlist[ylen+ulen:ylen+ulen+vlen]
        return y_dcode, u_dcode, v_dcode
        pass

    def __Zag(self, dcode):
        dcode = np.array(dcode).reshape((len(dcode) // 64, 64))
        tz = np.zeros(dcode.shape)
        for i in range(len(self.__zag)):
            tz[:, i] = dcode[:, self.__zag[i]]
        rlist = tz.tolist()
        return rlist

    def __IRle(self, dcode):
        rlist = []
        for i in range(len(dcode)):
            if i % 2 == 0:
                rlist += [0] * dcode[i]
            else:
                rlist.append(dcode[i])
        return rlist

    def Decompress(self, filename):
        y_dcode, u_dcode, v_dcode = self.__ReadFile(filename)
        y_blocks = self.__Zag(self.__IRle(y_dcode))
        u_blocks = self.__Zag(self.__IRle(u_dcode))
        v_blocks = self.__Zag(self.__IRle(v_dcode))
        y = self.__Decode(y_blocks, self.__lt)
        u = self.__Decode(u_blocks, self.__ct)
        v = self.__Decode(v_blocks, self.__ct)
        r = (y + 1.402 * (v - 128))
        g = (y - 0.34414 * (u - 128) - 0.71414 * (v - 128))
        b = (y + 1.772 * (u - 128))
        r = Image.fromarray(r).convert('L')
        g = Image.fromarray(g).convert('L')
        b = Image.fromarray(b).convert('L')
        image = Image.merge("RGB", (r, g, b))
        image.save("./filename.bmp", "bmp")
        print(f"save to {filename}.bmp in file.")
        image.show()

    def __VLI(self, n):
        # Gets the variable word length integer encoding of integer n
        ts, tl = 0, 0
        if n > 0:
            ts = bin(n)[2:]
            tl = len(ts)
        elif n < 0:
            tn = (-n) ^ 0xFFFF
            tl = len(bin(-n)[2:])
            ts = bin(tn)[-tl:]
        else:
            tl = 0
            ts = '0'
        return (tl, ts)

    def __IVLI(self, tl, ts):
        # Obtaining the integer n corresponding to the integer code with variable word length
        if tl != 0:
            n = int(ts, 2)
            if ts[0] == '0':
                n = n ^ 0xFFFF
                n = int(bin(n)[-tl:], 2)
                n = -n
        else:
            n = 0
        return n

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]  # Get the file name from the command line
    kjpeg = KJPEG()
    kjpeg.Compress(filename)
    gpj = filename.replace('.jpg', '.gpj')
    kjpeg.Decompress(gpj)
    os.remove(gpj)
