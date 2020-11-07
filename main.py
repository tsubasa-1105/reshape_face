import cv2,glob,os,sys
import numpy as np

class Image():

    # コンストラクタ
    def __init__(self,file):

        if type(file) == np.ndarray:
            self.data = file
        if type(file) == str:
            self.data = cv2.imread(file)

    # 画像を保存
    def save(self,file:str='tmp.png'):
        return cv2.imwrite(file,self.data)

    # 画像を表示
    def show(self,msg:str='Debug'):
        cv2.imshow(msg,self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 画像をリサイズ
    def resize(self,width:int,height:int):
        self.data = cv2.resize(self.data,dsize=(width,height))

    # 画像をトリミング
    def clipped(self,rect:tuple):
        if len(rect) != 4:
            print('clipped :','rect takes from 4 arguments but',len(rect),'were given.')
            return tuple()
        self.data = self.data[rect[1]:rect[3],rect[0]:rect[2],:]

    # 画像サイズを取得
    def shape(self):
        h,w,_ = self.data.shape[:3]
        return w,h

    # 顔検出
    def get_face(self):
        # カラー画像
        img = self.data
        # グレースケールに変換
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # カスケード検出器の特徴量を取得する
        cascade = cv2.CascadeClassifier('face.xml')
        # 顔検出の実行
        facerect = cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=2,minSize=(50, 50))
        # 一番面積が大きい顔の矩形を取得
        return self.__get_most_rect(facerect)

    # 一番面積が大きい矩形を返す
    def __get_most_rect(self,facerect)->tuple:
        res = None
        if len(facerect) > 0:
            for rect in facerect:
                p1 = tuple(rect[0:2])
                p2 = tuple(rect[0:2]+rect[2:4])
                area = (p2[1]-p1[1])*(p2[0]-p1[0])
                if res == None:
                    res = [area,p1,p2]
                else:
                    if res[0] < area:
                        res = [area,p1,p2]

            #cv2.rectangle(self.data,res[1],res[2],(0,255,0),thickness=1)
            return self.rect_to_point(res[1]+res[2])
        else:
            return res
    
    # 中心座標を計算する
    def rect_to_point(self,rect:tuple)->tuple:
        if len(rect) != 4:
            print('rect_to_point :','rect takes from 4 arguments but',len(rect),'were given.')
            return tuple()
        else:
            x = int((rect[2]+rect[0])/2)
            y = int((rect[3]+rect[1])/2)
            return (x,y)
    
    # 顔を考慮した切り抜き矩形を求める
    def get_square_by_consider_face(self):

        # 画像の中心座標を求める
        w,h = self.shape()
        c = self.rect_to_point((0,0,w,h))
        # 矩形幅の設定
        size = int((min(w,h))/2)
        # 切り取る矩形範囲を計算
        face = self.get_face()
        #cv2.drawMarker(self.data,face,(0,0,255))

        if face != None:  # 顔が検出された場合
            tmp = [face[0]-size,face[1]-size,face[0]+size,face[1]+size]
            if tmp[0] < 0:
                tmp[2] += (tmp[0]*-1)
                tmp[0] += (tmp[0]*-1)
            if tmp[1] < 0:
                tmp[3] += (tmp[1]*-1)
                tmp[1] += (tmp[1]*-1)
            if tmp[2] > w:
                tmp[0] -= tmp[2]-w
                tmp[2] -= tmp[2]-w
            if tmp[3] > h:
                tmp[1] -= tmp[3]-h
                tmp[3] -= tmp[3]-h
            rect = tuple(tmp)
        else:              # 顔が検出されなかった場合
            rect = (c[0]-size,c[1]-size,c[0]+size,c[1]+size)
        
        return rect

    # 枠線を付ける
    def put_border(self,color,thickness=3):
        cv2.rectangle(img.data,(0,0),img.shape(),color,thickness)

# 動画から画像を切り出す
def get_image_for_video(path):
    cap = cv2.VideoCapture(path)
    _,frame = cap.read()
    return Image(frame)

if __name__ == "__main__":

    path = glob.glob('video/*.mp4')
    for w in path:

        # ファイル名を取得
        name = os.path.basename(w)
        # 保存先を指定
        dst = 'dst/'+os.path.splitext(os.path.basename(name))[0]+'.png'
        # 動画を画像に変換
        img = get_image_for_video(w)

        # 切り抜く正方形を取得
        rect = img.get_square_by_consider_face()
        # 画像から矩形を切り抜く
        img.clipped(rect)
        # 画像をリサイズ
        img.resize(200,200)
        # 枠線を付ける
        img.put_border(color=(84,84,81))
        # ファイルを保存
        res = img.save((dst))
        print('',name,'->','成功' if res else '失敗')
        # 結果を表示
        # img.show()
