import sys
import math
import numpy as np
from math import cos, pi, sin
from PyQt5.QtMultimedia import QMediaPlaylist, QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QDirIterator, Qt
from PyQt5.QtWidgets import (QAction, QApplication, QGridLayout, QLabel,
        QMainWindow, QMessageBox, QOpenGLWidget, QScrollArea,
        QSizePolicy, QSlider, QWidget)


import OpenGL.GL as gl
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
from OpenGL.GLU import *

from data.read_motion_utils.animation_process import *


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        for coord in ('x', 'y', 'z', 'cx', 'cy', 'cz', 'rx', 'ry', 'rz'):
            setattr(self, coord, 50 if coord == 'z' else 0)



        sps_path = "./data/motion_music_align_dance1anddance2/motion_align/dance2_1_01_faceZ.sps"
        music_path = "./data/motion_music_align_dance1anddance2/music_align/xiaoxinyun1_01.wav"

        dance_class = "modern"
        mskel_name = "xiaowen" if dance_class == "modern" else "xiaoyou"

        self.bone_motion = get_motion_bone_position(sps_path, mskel_name)

        #-----for music
        self.player = QMediaPlayer()
        self.playlist = QMediaPlaylist()
        url = QUrl.fromLocalFile(music_path)
        self.playlist.addMedia(QMediaContent(url))
        self.player.setVolume(100)
        self.player.setPlaylist(self.playlist)
        self.frame = 0


    def SetupRC(self):
        # Light values and coordinates光照 值与坐标；环境光，漫射光，镜面光，光的坐标,
        ambientLight = [0.4, 0.4, 0.4, 1.0]
        diffuseLight = [0.7, 0.7, 0.7, 1.0]
        specular = [0.9, 0.9, 0.9, 1.0]
        lightPos = [-50.0, 200.0, 200.0, 1.0]
        specref = [0.6, 0.6, 0.6, 1.0]

        glEnable(GL_DEPTH_TEST)  # Hidden surface removal
        glEnable(GL_CULL_FACE)  # Do not calculate inside of solid object
        glFrontFace(GL_CCW)

        glEnable(GL_LIGHTING)

        # Setup light 0
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular)

        # Position and turn on the light
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
        glEnable(GL_LIGHT0)

        # Enable color tracking
        glEnable(GL_COLOR_MATERIAL)

        # Set Material properties to follow glColor values
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

        # All materials hereafter have full specular reflectivity with a moderate shine
        glMaterialfv(GL_FRONT, GL_SPECULAR, specref)
        glMateriali(GL_FRONT, GL_SHININESS, 64)

        glClearColor(1.0, 1.0, 1.0, 0.0)

    def initializeGL(self):
        self.SetupRC()
        glMatrixMode(GL_PROJECTION)

    def paintGL(self):
        # -----for music
        self.player.play()


        self.drawGrid()
        self.draw_one_frame()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(self.x, self.y + 5, self.z, self.cx, self.cy, self.cz, 0, 1, 0)
        self.update()



    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def wheelEvent(self, event):
        self.z += -2 if event.angleDelta().y() > 0 else 2

    def mouseMoveEvent(self, event):
        dx, dy = event.x() - self.last_pos.x(), event.y() - self.last_pos.y()
        if event.buttons() == Qt.LeftButton:
            self.x, self.y = self.x - 0.1 * dx, self.y + 0.1 * dy
        elif event.buttons() == Qt.RightButton:
            self.cx, self.cy = self.cx + dx / 50, self.cy + dy / 50
        self.last_pos = event.pos()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.timer.isActive():
                self.timer.stop()
            else:
                self.timer.start()

    def get_pose_frame(self):
        self.frame += 1

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport((width - side) // 2, (height - side) // 2, side, side)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glFrustum(-3.0, +3.0, -1.5, 1.5, 5.0, 60.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslated(0.0, 0.0, -100.0)




    def drawGrid(self):
        gl.glLineWidth(1.0)
        gl.glColor4f(0.4, 0.4, 0.4, 0.5)
        gl.glBegin(gl.GL_LINES)
        g_grid_size = 1.0
        g_unit = 1.0

        for i in range(-15, 16):
            if i == 0:
                continue
            gl.glVertex3f( -15.0 * g_grid_size * g_unit, 0.0, i * g_grid_size * g_unit)
            gl.glVertex3f( 15.0 * g_grid_size * g_unit, 0.0, i * g_grid_size * g_unit)
        for i in range(-15, 16):
            if i == 0:
                continue
            gl.glVertex3f(i * g_grid_size * g_unit, 0.0, -15.0 * g_grid_size * g_unit)
            gl.glVertex3f(i * g_grid_size * g_unit, 0.0, 15.0 * g_grid_size * g_unit)
        gl.glEnd()
        gl.glLineWidth(2.5)

        gl.glColor4f(0.5, 0.5, 1.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(-15.0 * g_grid_size * g_unit, 0.0, 0.0)
        gl.glVertex3f(15.0 * g_grid_size * g_unit, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, -15.0 * g_grid_size * g_unit)
        gl.glVertex3f(0.0, 0.0, 15.0 * g_grid_size * g_unit)
        gl.glEnd()



    def draw_one_frame(self):
        self.frame = int(self.player.position() * 60. / 1000.)

        if self.frame < 0:
            self.frame = 0
        if self.frame >= self.bone_motion.pose_num:
            self.frame = self.bone_motion.pose_num - 1

        bones_position, bones_pair = self.bone_motion.get_sps_motion(self.frame)

        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GL_SMOOTH)
        gluQuadricTexture(quadric, GL_TRUE)
        init_vec = np.array([0.0, 0.0, 1.0])
        gl.glColor4f(0.9, 0.3, 0.3, 1.0)
        for i, b_p in enumerate(bones_pair):
            glPushMatrix()
            gl.glTranslatef(0.005*b_p[1][0], 0.005*b_p[1][1], 0.005*b_p[1][2])
            x = b_p[0][0] - b_p[1][0]
            y = b_p[0][1] - b_p[1][1]
            z = b_p[0][2] - b_p[1][2]
            norm = 0.005 * np.sqrt(x*x+y*y+z*z)
            link_transform = QMatrix4x4()
            link_transform.setToIdentity()
            direction = np.array([x, y, z])
            angle = math.acos(init_vec.dot(direction) / (np.sqrt(np.sum(init_vec ** 2))
                                                             * np.sqrt(np.sum(direction ** 2)) + 1e-6))
            axis = np.cross(init_vec, direction) / (np.sqrt(np.sum(np.cross(init_vec, direction) ** 2)) + 1e-6)
            r = QQuaternion().fromAxisAndAngle(QVector3D(axis[0], axis[1], axis[2]), radian2degree(angle))
            link_transform.rotate(r)
            rot_m = np.mat(np.array(link_transform.data()).reshape(4,4))
            glMultMatrixf(rot_m)
            if 16 <= i <= 39 or 44 <=i < 68:
                k = 0.07
            else:
                k= 0.2
            if norm > 0.07:
                gluCylinder(quadric, k, k, norm, 30, 30)

            gluSphere(quadric, k, 30, 30)


            glPopMatrix()


class Scene_MainWindow(QMainWindow):
    def __init__(self):
        super(Scene_MainWindow, self).__init__()

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        self.glWidget = GLWidget()
        self.pixmapLabel = QLabel()

        self.glWidgetArea = QScrollArea()
        self.glWidgetArea.setWidget(self.glWidget)
        self.glWidgetArea.setWidgetResizable(True)
        self.glWidgetArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.glWidgetArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.glWidgetArea.setSizePolicy(QSizePolicy.Ignored,
                QSizePolicy.Ignored)
        self.glWidgetArea.setMinimumSize(500, 500)



        self.createActions()
        self.createMenus()

        centralLayout = QGridLayout()
        centralLayout.addWidget(self.glWidgetArea, 0, 0)
        centralWidget.setLayout(centralLayout)



        self.setWindowTitle("Music2dance")
        self.resize(1600, 1200)




    def about(self):
        QMessageBox.about(self, "About Grabber",
                "The <b>Grabber</b> example demonstrates two approaches for "
                "rendering OpenGL into a Qt pixmap.")

    def createActions(self):
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

    def createSlider(self, changedSignal, setterSlot):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 360 * 16)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setTickPosition(QSlider.TicksRight)

        slider.valueChanged.connect(setterSlot)
        changedSignal.connect(slider.setValue)

        return slider




if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainWin = Scene_MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
