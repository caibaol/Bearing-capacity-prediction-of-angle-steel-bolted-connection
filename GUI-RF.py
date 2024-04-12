import sys
import sip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from plot_frame import plot_frame
from matplotlib import rcParams, ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QComboBox, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, Qt, QSize
from PyQt5 import QtGui




config = {
    "font.family":'Arial',
    "font.size": 15,
    # "mathtext.fontset":'stix',
}
rcParams.update(config)
filename = 'result_RF.sav'
# gm    = np.loadtxt('./GroundMotion.txt')
model = pickle.load(open(filename ,'rb'))


#print(gm)
dpi   = 170
font_size_label = 8
font_size_tick  = 8


class predictor(QWidget):

    def __init__(self):
        super(predictor, self).__init__()
        self.initUI()
        self.setWindowTitle('The bearing capacity of angle steel bolt connection ')
        
        #self.setWindowIcon(QtGui.QIcon(sys.path[0]+'/logo.jpg'))

    def initUI(self):
        # subtitle
        self.intro_label = QLabel('The bearing capacity of angle steel bolt connection ')
        self.intro_label.setAlignment(Qt.AlignCenter)
        self.intro_label.setStyleSheet('color:rgb(0,150,195); font-weight:bold; background-color:orange; border-radius:10px; border:2px groove gray; border-style:outset')
        self.intro_label.setFixedHeight(50)

        event_name_list = ['2016-02-06-Meinong-Taiwan-China', '2022-02-06-Meinong-Taiwan-China']
        # parameters
        self.ns_label  = QLabel('Yield Strength of material-Fu(MPa)', self)
        self.hs_label  = QLabel('Strength of material-Fy(MPa)', self)
        self.nss_label = QLabel('Number of bolts', self)
        self.lss_label = QLabel('Width of angle steel(mm)', self)
        self.nls_label = QLabel('Thickness of angle steel(mm)', self)
        self.lls_label = QLabel('Bolt diameter(mm)', self)
        self.cp_label  = QLabel('Diameter of bolt hole(mm) ', self)
        self.sdi_label = QLabel('Bolt spacing distance-p1(mm)', self)
        self.lat_label = QLabel('Bolt edge distance-e2(mm)', self)
        self.lon_label = QLabel('Bolt end distance-e1(mm)', self)
        self.prob_label = QLabel('Fbc(kN)', self)
        # input box
        self.ns_line = QLineEdit('300', self)
        self.ns_line.setStyleSheet('background-color:white; border-radius:10px')
        # self.ns_line.setFixedWidth(323)
        self.ns_line.setFixedWidth(210)
        # self.ns_line.editingFinished.connect(self.on_frame_change)
        self.hs_line     = QLineEdit('459.1', self)
        self.hs_line.setStyleSheet('background-color:white; border-radius:10px')
        #self.hs_line.editingFinished.connect(self.on_frame_change)
        self.nss_line    = QLineEdit('1', self)
        self.nss_line.setStyleSheet('background-color:white; border-radius:10px')
        #self.nss_line.editingFinished.connect(self.on_frame_change)
        self.lss_line    = QLineEdit('56.19', self)
        self.lss_line.setStyleSheet('background-color:white; border-radius:10px')
        #self.lss_line.editingFinished.connect(self.on_frame_change)
        self.nls_line     = QLineEdit('4', self)
        self.nls_line.setStyleSheet('background-color:white; border-radius:10px')
        #self.nls_line.editingFinished.connect(self.on_frame_change)
        self.lls_line = QLineEdit('16', self)
        self.lls_line.setStyleSheet('background-color:white; border-radius:10px')
        #self.lls_line.editingFinished.connect(self.on_frame_change)
        self.cp_line  = QLineEdit('17.5', self)
        self.cp_line.setStyleSheet('background-color:white; border-radius:10px')
        self.sdi_line  = QLineEdit('56', self)
        self.sdi_line.setStyleSheet('background-color:white; border-radius:10px')
        self.lat_line          = QLineEdit('28', self)
        self.lat_line.setStyleSheet('background-color:white; border-radius:10px')
        self.lon_line          = QLineEdit('25', self)
        self.lon_line.setStyleSheet('background-color:white; border-radius:10px')
        self.ds_line   = QLineEdit(self)
        self.ds_line.setStyleSheet('background-color:white; border-radius:10px')
        self.prob_line       = QLineEdit(self)
        self.prob_line.setStyleSheet('background-color:white; border-radius:10px')

        # button
        self.pred_button = QPushButton('Predict', self)
        self.pred_button.clicked.connect(self.on_pred_button_click)
        self.pred_button.setStyleSheet('color:red; background-color:rgb(0,150,195); border-radius:10px; border:2px groove gray; border-style:outset;')

        

        # layout
        self.grid_layout_intro  = QGridLayout()
        self.grid_layout_build  = QGridLayout()
        #self.grid_layout_event = QGridLayout()
        self.grid_layout_output   = QGridLayout()
        self.grid_layout_pred   = QGridLayout()
        #self.grid_layout_frame   = QGridLayout()
        #self.grid_layout_gm_fig   = QGridLayout()
        #self.grid_layout_prob   = QGridLayout()
        
        spacing = 23
        self.grid_layout_intro.addWidget(self.intro_label, 0, 0, 1, 3)
        self.row_start_build = 2
        self.grid_layout_build.addWidget(self.ns_label, self.row_start_build+0, 0, 1, 1)
        self.grid_layout_build.addWidget(self.ns_line, self.row_start_build+0, 1, 1, 1)
        self.grid_layout_build.addWidget(self.hs_label, self.row_start_build+1, 0, 1, 1)
        self.grid_layout_build.addWidget(self.hs_line, self.row_start_build+1, 1, 1, 1)
        self.grid_layout_build.addWidget(self.nss_label, self.row_start_build+2, 0, 1, 1)
        self.grid_layout_build.addWidget(self.nss_line, self.row_start_build+2, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lss_label, self.row_start_build+3, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lss_line, self.row_start_build+3, 1, 1, 1)
        self.grid_layout_build.addWidget(self.nls_label, self.row_start_build+4, 0, 1, 1)
        self.grid_layout_build.addWidget(self.nls_line, self.row_start_build+4, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lls_label, self.row_start_build+5, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lls_line, self.row_start_build+5, 1, 1, 1)
        self.grid_layout_build.addWidget(self.cp_label, self.row_start_build+6, 0, 1, 1)
        self.grid_layout_build.addWidget(self.cp_line, self.row_start_build+6, 1, 1, 1)
        self.grid_layout_build.addWidget(self.sdi_label, self.row_start_build+7, 0, 1, 1)
        self.grid_layout_build.addWidget(self.sdi_line, self.row_start_build+7, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lat_label, self.row_start_build+8, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lat_line, self.row_start_build+8, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lon_label, self.row_start_build+9, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lon_line, self.row_start_build+9, 1, 1, 1)
        self.grid_layout_build.setSpacing(spacing)
        
        self.row_start_output = 14
        self.grid_layout_output.addWidget(self.prob_label, self.row_start_output+1, 0, 1, 1)
        self.grid_layout_output.addWidget(self.prob_line, self.row_start_output+1, 1, 1, 1)
        self.grid_layout_output.setSpacing(spacing)
        self.row_start_pred = 16
        self.grid_layout_pred.addWidget(self.pred_button, self.row_start_pred+0, 0, 1, 2)
        

        self.groupbox_build  = QGroupBox('Input-specimen information', self)
        self.groupbox_build.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_build.setLayout(self.grid_layout_build)
        
        self.groupbox_output = QGroupBox('Output-bearing capacity', self)
        self.groupbox_output.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_output.setLayout(self.grid_layout_output)

        self.vbox_layout_var = QVBoxLayout()
        self.vbox_layout_plot = QVBoxLayout()
        self.hbox_layout_var_plot = QHBoxLayout()
        self.vbox_layout_all = QVBoxLayout()
        self.vbox_layout_var.addWidget(self.groupbox_build)
        
        self.vbox_layout_var.addWidget(self.groupbox_output)
        self.vbox_layout_var.addLayout(self.grid_layout_pred)

        self.hbox_layout_var_plot.addLayout(self.vbox_layout_var)
        self.hbox_layout_var_plot.addLayout(self.vbox_layout_plot)
        self.vbox_layout_all.addLayout(self.grid_layout_intro)
        self.vbox_layout_all.addLayout(self.hbox_layout_var_plot)

        self.setLayout(self.vbox_layout_all)
    


    @pyqtSlot()
    def on_pred_button_click(self):
        self.ns_val  = float(self.ns_line.text())
        self.hs_val  = float(self.hs_line.text())
        self.nss_val = float(self.nss_line.text())
        self.lss_val = float(self.lss_line.text())
        self.nls_val = float(self.nls_line.text())
        self.lls_val = float(self.lls_line.text())
        self.cp_val  = float(self.cp_line.text())
        self.sdi_val = float(self.sdi_line.text())
        self.lat_val = float(self.lat_line.text())
        self.lon_val = float(self.lon_line.text())

        # predict damage state
        features= [np.array([self.ns_val, self.hs_val, self.nss_val, self.lss_val, self.nls_val, self.lls_val,self.cp_val,self.sdi_val,self.lat_val,self.lon_val,])]
        pred          = model.predict(features)
        output        = float(pred)
        # self.prob_line = QLineEdit(output, self)
        #self.prob_val = float(self.prob_line.text())
        self.prob_line.setText('{0:.2f}'.format(output))
        #self.prob_line.QLineEdit('output', self)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QtGui.QFont()
    # font.setFamily("Times New Roman") # 字体
    font.setFamily("Arial") # 字体
    font.setPointSize(15)   # 字体大小
    app.setFont(font)
    demo = predictor()
    demo.show()
    sys.exit(app.exec_())