# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


import os
import numpy as np
import time
import copy

from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

from src.Ui_StimulateDialog import Ui_Stimulate
from src.infor_com_mea.spike_detection import SpikeDetection 

class StimulateDialog(QDialog, Ui_Stimulate):
    close_widget = pyqtSignal(bool)

    def __init__(self, parent=None, para=None, spike_para=None, recording=None, stimulating=None):
        super(StimulateDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.para = para    # 刺激参数
        self.spike_para = spike_para
        self.recording = recording
        self.stimulating = stimulating

        self.start_mark = 0         # 标记一次刺激开始
        self.sti_frame_list = []    # 记录一次刺激的刺激frame
        self.last_time_once_sti = self.para.sti_para["total_time"] / 1000 # 一次刺激总时长,s
        self.sti_frequency = self.para.para[0]["frequency"]
        self.one_mean_map = copy.copy(self.recording.channel_map)
        for i in self.one_mean_map:
            self.one_mean_map[i] = []

        if self.para is not None:
            self.lineEdit_stimulating_mode.setText(self.para.sti_para["stimulate_name"])

        if self.spb_start_time.value() >= self.spb_end_time.value():
            self.spb_start_time.setValue(self.spb_end_time.value() - 10)

        self.start_end_change(0)

        self.init_pb()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.get_recording_data_after_once_sti)

        self.spike_detection_thread = SpikeDetection(
            sample_rate=spike_para["Sample_Rate"], order=spike_para["Filter_Order"],
            min_fre=spike_para["Min_Frequency"], max_fre=spike_para["Max_Frequency"], 
            multiplier=spike_para["Multiplier"], period=spike_para["Refractor_Time"])
        self.spike_detection_thread.start()

        self.pb_stimulate.clicked.connect(self.stimulate)
        self.pb_close.clicked.connect(self.close_wd)

        self.spb_start_time.valueChanged.connect(self.start_end_change)
        self.spb_end_time.valueChanged.connect(self.start_end_change)

    def init_pb(self):
        self.frame_array.setStyleSheet(
                "QFrame{background-color: rgb(250, 250, 250);}\n"
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(220, 220, 220);\n"
                "border-width:2px;\n"
                "border-style:soild;\n"
                "}\n"
                "\n"
                "QPushButton:hover{\n"
                "color:white;\n"
                "border-radius:25px;\n"
                "background-color:gray;\n"
                "border-color:rgb(200, 200, 200);\n"
                "}\n"
                "\n"
                "QPushButton:pressed{\n"
                "color:black;\n"
                "border-radius:25px;\n"
                "background-color: rgb(173, 255, 172);\n"
                "}\n"
                "")

        self.pb_button = {"21":self.pb_21,"31":self.pb_31,"41":self.pb_41,
                            "51":self.pb_51,"61":self.pb_61,"71":self.pb_71,

                "12":self.pb_12,"22":self.pb_22,"32":self.pb_32,"42":self.pb_42,
                "52":self.pb_52,"62":self.pb_62,"72":self.pb_72,"82":self.pb_82,

                "13":self.pb_13,"23":self.pb_23,"33":self.pb_33,"43":self.pb_43,
                "53":self.pb_53,"63":self.pb_63,"73":self.pb_73,"83":self.pb_83,
                
                "14":self.pb_14,"24":self.pb_24,"34":self.pb_34,"44":self.pb_44,
                "54":self.pb_54,"64":self.pb_64,"74":self.pb_74,"84":self.pb_84,
                
                "25":self.pb_25,"35":self.pb_35,"45":self.pb_45,"55":self.pb_55,
                "65":self.pb_65,"75":self.pb_75,"85":self.pb_85,
                
                "16":self.pb_16,"26":self.pb_26,"36":self.pb_36,"46":self.pb_46,
                "56":self.pb_56,"66":self.pb_66,"76":self.pb_76,"86":self.pb_86,
                
                "17":self.pb_17,"27":self.pb_27,"37":self.pb_37,"47":self.pb_47,
                "57":self.pb_57,"67":self.pb_67,"77":self.pb_77,"87":self.pb_87,
                
                "28":self.pb_28,"38":self.pb_38,"48":self.pb_48,
                "58":self.pb_58,"68":self.pb_68,"78":self.pb_78}

        for i in self.pb_button:
            self.pb_button[i].setText("0")
            self.comb_electrode.addItem(i)
        self.comb_electrode.currentIndexChanged.connect(self.combox_electrode)
        cur = self.comb_electrode.currentText()
        self.cur_electrode_key = cur    
        
        self.Jet = [[1.0,1.0,1.0],
           [0.99685320,1.00000000,0.99608661],
           [0.98882479,0.99728144,0.99353570],
           [0.98079638,0.99439511,0.99098479],
           [0.97276797,0.99150878,0.98843388],
           [0.96473956,0.98862245,0.98588297],
           [0.95671223,0.98574356,0.98334313],
           [0.94861712,0.98289474,0.98084241],
           [0.94044013,0.9800799 ,0.97838564],
           [0.93218602,0.97729661,0.97596977],
           [0.92384666,0.97454666,0.97359708],
           [0.91542029,0.97182969,0.97126719],
           [0.90691036,0.96914363,0.96897754],
           [0.89830971,0.96648977,0.96672992],
           [0.88961588,0.96386787,0.9645241 ],
           [0.88082645,0.9612776 ,0.96235983],
           [0.87194205,0.95871764,0.96023557],
           [0.8629679 ,0.95618273,0.95817252],
           [0.85397096,0.9536338 ,0.95631527],
           [0.84496294,0.95106327,0.95469334],
           [0.83595651,0.94846764,0.95329519],
           [0.82696415,0.945843  ,0.95211542],
           [0.81799269,0.9431889 ,0.95113448],
           [0.8090527 ,0.94050158,0.95035385],
           [0.80014507,0.9377831 ,0.94975033],
           [0.79127269,0.9350335 ,0.94931393],
           [0.78243744,0.93225265,0.94903955],
           [0.77363792,0.92944214,0.94891577],
           [0.76487124,0.92660389,0.94893194],
           [0.75613611,0.92373911,0.9490796 ],
           [0.74742686,0.92084986,0.94935405],
           [0.73874101,0.91793748,0.94974804],
           [0.73007248,0.91500396,0.95025777],
           [0.721417  ,0.91205082,0.9508781 ],
           [0.71277021,0.90907954,0.95160423],
           [0.70412657,0.90609115,0.9524379 ],
           [0.69548089,0.90308703,0.95337587],
           [0.68682749,0.900069  ,0.95441191],
           [0.67816176,0.89703816,0.95554255],
           [0.66947814,0.89399441,0.95677753],
           [0.66076997,0.89093979,0.95810849],
           [0.65203395,0.88787509,0.95952997],
           [0.64326339,0.88479964,0.96105698],
           [0.63445285,0.88171603,0.96267158],
           [0.62559603,0.87862293,0.96439183],
           [0.61668958,0.87552232,0.96619927],
           [0.60772508,0.87241305,0.9681133 ],
           [0.59869664,0.86929739,0.97011698],
           [0.58959903,0.86617315,0.97222978],
           [0.58042589,0.86304246,0.9744352 ],
           [0.5711695 ,0.85990457,0.97674364],
           [0.56178008,0.85677373,0.97911892],
           [0.55201392,0.85369958,0.98159081],
           [0.54179495,0.85070001,0.98410883],
           [0.53096   ,0.84779786,0.98672682],
           [0.5193188 ,0.84503107,0.98939076],
           [0.50649625,0.84247599,0.99200748],
           [0.4918648 ,0.84029536,0.99410799],
           [0.47567059,0.83865506,0.99366466],
           [0.4631307 ,0.83684342,0.98830818],
           [0.45445748,0.83455511,0.98069532],
           [0.44772827,0.83199804,0.97229723],
           [0.44205911,0.82928438,0.96360578],
           [0.4370532 ,0.82647088,0.95478082],
           [0.43254665,0.82358837,0.94582886],
           [0.42833298,0.82065567,0.93690996],
           [0.42440007,0.81768565,0.92794059],
           [0.42063242,0.81468792,0.91901487],
           [0.4169943 ,0.81166934,0.91013026],
           [0.41345657,0.8086362 ,0.90127781],
           [0.40999348,0.80559121,0.89247378],
           [0.40658436,0.80253923,0.8837073 ],
           [0.40321058,0.79948299,0.87498248],
           [0.39985589,0.79642487,0.86630251],
           [0.39650581,0.79336675,0.85767195],
           [0.39314806,0.79031081,0.84908926],
           [0.38977115,0.78725822,0.8405603 ],
           [0.38636568,0.78421086,0.83208114],
           [0.38292289,0.78116987,0.82365318],
           [0.37943522,0.77813629,0.81527678],
           [0.37592101,0.77511237,0.80690146],
           [0.37234859,0.77209714,0.79858211],
           [0.36871275,0.769092  ,0.79031112],
           [0.36501021,0.76609815,0.78207955],
           [0.36126092,0.76311598,0.77385056],
           [0.35743415,0.76014614,0.76566128],
           [0.35352653,0.7571879 ,0.75752052],
           [0.34955704,0.75424405,0.74936832],
           [0.34550548,0.75131274,0.74124954],
           [0.34136355,0.74839463,0.7331657 ],
           [0.33715355,0.74549121,0.72506347],
           [0.33285042,0.74260132,0.71698707],
           [0.32845699,0.73972439,0.70893287],
           [0.32398311,0.73686303,0.70085313],
           [0.31940857,0.7340142 ,0.69280344],
           [0.31474955,0.73117983,0.68473287],
           [0.30999283,0.72835863,0.6766676 ],
           [0.30513826,0.72555079,0.66859838],
           [0.30019124,0.72275692,0.66050262],
           [0.29514001,0.71997319,0.65243126],
           [0.28999609,0.71720523,0.64429651],
           [0.28474351,0.7144467 ,0.63618265],
           [0.27939297,0.71170224,0.62801219],
           [0.27393583,0.70896603,0.61985318],
           [0.26837081,0.70624357,0.61163369],
           [0.26274229,0.7035238 ,0.603409  ],
           [0.25708918,0.70080317,0.59516032],
           [0.25139236,0.69808446,0.58687536],
           [0.24565171,0.69536781,0.57854521],
           [0.23991149,0.69264627,0.57018865],
           [0.23410028,0.6899301 ,0.56176922],
           [0.2283514 ,0.68720004,0.55334425],
           [0.22251874,0.68447718,0.54483691],
           [0.2167686 ,0.68173853,0.53631837],
           [0.21098828,0.67899943,0.52773537],
           [0.20523844,0.6762522 ,0.51910608],
           [0.19956277,0.67349207,0.51043968],
           [0.19386978,0.6707304 ,0.50169604],
           [0.18834464,0.66794594,0.49293847],
           [0.18286164,0.66515446,0.48410892],
           [0.17746015,0.66235172,0.47521957],
           [0.17227677,0.65952486,0.46630039],
           [0.16719126,0.65668718,0.45731071],
           [0.16228961,0.65383203,0.44825488],
           [0.15767806,0.65094919,0.43917857],
           [0.15326739,0.64805094,0.43001356],
           [0.14911371,0.64513211,0.42078715],
           [0.14540024,0.64217848,0.41153616],
           [0.14201515,0.63920233,0.40221151],
           [0.13900491,0.63620109,0.39281699],
           [0.13647991,0.63316774,0.38335989],
           [0.13451467,0.63009637,0.37387115],
           [0.13306038,0.62699291,0.36431428],
           [0.13216412,0.62385465,0.35468331],
           [0.1318636 ,0.6206785 ,0.34498184],
           [0.13222338,0.61745798,0.33523319],
           [0.13327234,0.61418927,0.32544266],
           [0.13496383,0.61087421,0.31559078],
           [0.13730518,0.60750927,0.30567678],
           [0.14028754,0.60409089,0.29570821],
           [0.1438988 ,0.60061531,0.28568814],
           [0.1481163 ,0.59707868,0.27562355],
           [0.15292056,0.59347667,0.26551506],
           [0.15827476,0.58980531,0.255375  ],
           [0.1641346 ,0.58606088,0.24521969],
           [0.17047768,0.58223829,0.23504914],
           [0.17724568,0.57833448,0.22489109],
           [0.18440836,0.57434475,0.21475475],
           [0.19194283,0.570261  ,0.2047188 ],
           [0.1997894 ,0.56608181,0.19480014],
           [0.20786987,0.56180882,0.18501516],
           [0.21611525,0.55744299,0.17541354],
           [0.2245112 ,0.55297678,0.16613439],
           [0.23294787,0.54842038,0.15719997],
           [0.24132915,0.54378463,0.14864229],
           [0.24962913,0.53906807,0.14064234],
           [0.2577352 ,0.53428995,0.13319152],
           [0.26556953,0.5294661 ,0.12631077],
           [0.27312892,0.5245973 ,0.12018481],
           [0.28032555,0.51970779,0.11470778],
           [0.28713321,0.51480949,0.10988471],
           [0.29356587,0.50990519,0.1057921 ],
           [0.2996058 ,0.50500706,0.10236765],
           [0.30524535,0.50012545,0.09954285],
           [0.31050794,0.49526299,0.09728316],
           [0.31540309,0.49042474,0.09554714],
           [0.3199562 ,0.48561167,0.09429097],
           [0.32417869,0.48082736,0.09346674],
           [0.32810158,0.47606977,0.09304063],
           [0.33173453,0.47134236,0.09293914],
           [0.33509073,0.46664674,0.09311555],
           [0.33819806,0.4619795 ,0.09355485],
           [0.34106783,0.45734189,0.09420636],
           [0.34371655,0.45273312,0.09503429],
           [0.34616418,0.44815082,0.09601148],
           [0.34846161,0.44358741,0.09694708],
           [0.35064144,0.4390366 ,0.09777621],
           [0.35270902,0.43449794,0.09850264],
           [0.35466885,0.42997113,0.09912939],
           [0.35652668,0.42545536,0.09966036],
           [0.35828593,0.42095047,0.10009768],
           [0.35995007,0.41645625,0.10044284],
           [0.36152319,0.41197217,0.10069866],
           [0.36301095,0.40749701,0.10086947],
           [0.3644172 ,0.40303012,0.10095768],
           [0.36574486,0.3985711 ,0.10096505],
           [0.36699958,0.39411845,0.1008952 ],
           [0.36818239,0.38967232,0.10074873],
           [0.36929692,0.3852318 ,0.10052801],
           [0.37034512,0.38079658,0.10023439],
           [0.37133204,0.37636503,0.09987124],
           [0.37225908,0.37193689,0.09943971],
           [0.37312827,0.3675116 ,0.09894146],
           [0.37394302,0.36308796,0.09837903],
           [0.37470512,0.35866538,0.09775384],
           [0.37541289,0.35424476,0.09706461],
           [0.37607367,0.34982294,0.09631749],
           [0.37668603,0.34540057,0.09551274],
           [0.37725529,0.3409752 ,0.09465424],
           [0.37778021,0.33654731,0.09374256],
           [0.37826417,0.33211517,0.09278067],
           [0.37870771,0.32767836,0.09176971],
           [0.37910605,0.32323907,0.09070563],
           [0.37946626,0.31879357,0.08959652],
           [0.37979138,0.31434   ,0.08844512],
           [0.38008051,0.30987844,0.08725167],
           [0.38032675,0.30541223,0.08600965],
           [0.38054033,0.30093554,0.0847296 ],
           [0.38072032,0.29644835,0.08341101],
           [0.38085917,0.29195449,0.08204562],
           [0.38096863,0.28744682,0.08064504],
           [0.38104043,0.28292961,0.07919895],
           [0.38108095,0.27839852,0.07771364],
           [0.38108465,0.27385632,0.07618048],
           [0.38106091,0.2692967 ,0.07460605],
           [0.38099775,0.26472644,0.07297407],
           [0.3809061 ,0.26013804,0.0712914 ],
           [0.38078597,0.25553073,0.06955101],
           [0.38063268,0.25090684,0.06773955],
           [0.38045132,0.24626226,0.06585239],
           [0.38024331,0.24159515,0.06387908],
           [0.38001011,0.23690363,0.06180541],
           [0.37975474,0.23218461,0.05961421],
           [0.37947787,0.22743677,0.05727841],
           [0.37917348,0.22266243,0.05479251],
           [0.37881491,0.21787271,0.05231823],
           [0.3783989 ,0.21306771,0.04989345],
           [0.37792763,0.20824511,0.04751508],
           [0.37739993,0.20340473,0.04518664],
           [0.37681735,0.19854439,0.04290621],
           [0.3761792 ,0.19366333,0.04067681],
           [0.37548684,0.18875918,0.03849184],
           [0.37473926,0.18383113,0.0364235 ],
           [0.37393841,0.17887592,0.0344685 ],
           [0.3730828 ,0.17389278,0.03262559],
           [0.37217413,0.16887809,0.03088785],
           [0.37121201,0.1638297 ,0.02925238],
           [0.37019649,0.15874474,0.02771554],
           [0.36912855,0.15361916,0.02627217],
           [0.36800743,0.14845004,0.02492019],
           [0.36683365,0.14323283,0.02365541],
           [0.36560768,0.13796238,0.02247376],
           [0.36432924,0.13263361,0.02137257],
           [0.36299804,0.12724067,0.02034929],
           [0.36161472,0.1217758 ,0.01939974],
           [0.36017918,0.11623077,0.018521  ],
           [0.35869134,0.11059595,0.01771024],
           [0.35715074,0.10486044,0.01696541],
           [0.35555893,0.09900824,0.01628386],
           [0.35392809,0.09300598,0.01563661],
           [0.35225838,0.08683204,0.01500064],
           [0.35054943,0.08045811,0.01437639],
           [0.34879865,0.07385239,0.01376336],
           [0.3470089 ,0.0669625 ,0.0131635 ],
           [0.34517814,0.0597292 ,0.01257656],
           [0.34330758,0.05206267,0.01200372],
           [0.34139609,0.04383912,0.01144512],
           [0.33944367,0.03500026,0.0109014 ],
           [0.3374549 ,0.02631803,0.0103749 ],
           [0.33546356,0.01784085,0.00987869]]
    
    def update_remain_time(self):
        re_time = self.last_time_once_sti - (time.time() - self.start_mark)
        self.spinBox_countdown.setValue(int(re_time))

    def stimulate(self):
        self.start_mark = time.time()
        self.pb_stimulate.setEnabled(False)
        self.pb_close.setEnabled(False)
        self.stimulating.update_stimulation_one_time([self.cur_electrode_key])
        self.recording.clear_recording_buffer(True)

        # 每次需要重置 临时map
        for i in self.one_mean_map:
            self.one_mean_map[i].clear()
            self.one_mean_map[i] = []

        # self.timer.start(self.spb_end_time.value() + 100)
        self.timer.start(1000)    # 每1s更新一次

    def get_recording_data_after_once_sti(self):  
        # if time.time() - self.start_mark > self.last_time_once_sti:
        self.timer.stop()
  
        electrode_key = self.recording.channel_map

        # t1 = time.time()  获取特定时间内的所有通道数据,channel_data在本次刺激中一直累加
        resolution, channel_data  = self.recording.get_target_time_signal()

        # 获得多段数据
        out_raw_data = self.get_spike_data_from_channel_data(resolution, channel_data)
        
        for i in electrode_key:    # 遍历所有通道，计算每个通道的spike
            if i == "15":
                continue
            index = electrode_key[i]

            # 获取
            spikes = []
            for j in range(len(out_raw_data)):
                y = out_raw_data[j]
                data = y[index - 1]  # channel 从0开始，需减1
                self.spike_detection_thread.set_data(data)
                self.spike_detection_thread.run()
                spike_num = self.spike_detection_thread.get_spike_num()
                spikes.append(spike_num)

            # 记录spike 以最后显示均值
            self.one_mean_map[i].append(np.sum(spikes))

            self.pb_button[i].setText(str(int(np.sum(spikes))))
            if len(spikes) > 0:
                # id = int(np.sum(spikes) * 3)
                id = int(np.sum(spikes))
            else:
                id = 0
            if id >= 255:
                r = int(self.Jet[255][0]*255)
                g = int(self.Jet[255][1]*255)
                b = int(self.Jet[255][2]*255)            
            else:
                r = int(self.Jet[id][0]*255)
                g = int(self.Jet[id][1]*255)
                b = int(self.Jet[id][2]*255) 
            self.pb_button[i].setStyleSheet("background-color: rgb(" + str(r)  + ", " + str(g) + "," + str(b) + ");")

        # print("time:", time.time() - t1)
        
        if time.time() - self.start_mark > self.last_time_once_sti:    # 最后显示各通道的总和spike个数
            self.recording.clear_recording_buffer(False)   # 不再记录数据
            self.spinBox_countdown.setValue(0)
            self.pb_stimulate.setEnabled(True)
            self.pb_close.setEnabled(True)
            self.timer.stop()

            # 重置标记的刺激frame列表
            self.sti_frame_list.clear()
            self.sti_frame_list = []

            for i in electrode_key:
                if i == "15":
                    continue

                spike_num = int(np.sum(self.one_mean_map[i]))
                self.pb_button[i].setText(str(spike_num))
                id = int(spike_num)
                if id >= 255:
                    r = int(self.Jet[255][0]*255)
                    g = int(self.Jet[255][1]*255)
                    b = int(self.Jet[255][2]*255)            
                else:
                    r = int(self.Jet[id][0]*255)
                    g = int(self.Jet[id][1]*255)
                    b = int(self.Jet[id][2]*255) 
                self.pb_button[i].setStyleSheet("background-color: rgb(" + str(r)  + ", " + str(g) + "," + str(b) + ");")
            
            self.recording.reset_target_time_process()    # 一次完整刺激结束后重置缓存池数据（清空）
            self.one_mean_map.clear()
            self.one_mean_map = None
        else:    # 信号未刺激完成，则重置缓存池，便于下一次读取和信号可视化
            self.update_remain_time()
            # self.recording.clear_recording_buffer(True)    不清除buffer
            self.timer.start()

    def get_spike_data_from_channel_data(self, resolution, channel_data):
        # 根据记录的刺激电极时刻，获取第一个刺激时刻信号,20帧内，不重复检测
        # 原69通道，可直接获取
        sti_signal = channel_data[69]

        # # 更改标识符之后，需检测出两个记录通道的刺激时刻，取并集
        # sti_sg1 = channel_data[70]
        # sti_sg2 = channel_data[71]
        # sti_signal = channel_data[69]    # 理论上全部为 0， 需测试一下
        # sti_signal[sti_sg1 > 0] = 1
        # sti_signal[sti_sg2 > 0] = 1      # 取两者并集


        non_zero = np.where(sti_signal > 0)[0]    # 此帧为首次刺激时刻，每个频率时刻，需要重置
        out_frame = []
        for i in range(len(non_zero)):
            if len(out_frame) == 0:
                out_frame.append(non_zero[i])
            elif non_zero[i] > out_frame[-1] + 20:    # 判断若20帧内，即400μs内2非刺激信号
                out_frame.append(non_zero[i])

        # 根据找出的检测位置，获得特定时间内的，目标区域的检测数据，用于spike检测
        out_raw_data = []    
        if len(out_frame) > 0:    # 有刺激时刻
            for i in range(len(out_frame)):
                cur = out_frame[i]    # 刺激起始位置
                if cur in self.sti_frame_list:
                    continue

                start = int(resolution * self.spb_start_time.value())    # 目标区域数据
                end = int(resolution * self.spb_end_time.value())
                if cur + end < len(channel_data[0]) - 1:    
                    spike_raw_data = channel_data[:, cur+start:cur+end]
                    out_raw_data.append(spike_raw_data)
                    self.sti_frame_list.append(cur)
        return out_raw_data



    def close_wd(self):
        self.close()
    
    def closeEvent(self, a0: QCloseEvent) -> None:
        self.stimulating.reset_electrode_stimulating_state()
        self.close_widget.emit(True)

        self.spike_detection_thread.quit()
        del self.spike_detection_thread
        self.spike_detection_thread = None
        return super().closeEvent(a0)
    
    def start_end_change(self, value):
        # 限制可设置的时间范围
        time_interval = int(1000.0 / self.sti_frequency)
        if self.spb_end_time.value() > time_interval:
            self.spb_end_time.setValue(time_interval)
            print("The time should be in the frequency interval!...")

        if self.spb_start_time.value() >= self.spb_end_time.value():
            self.spb_start_time.setValue(self.spb_end_time.value() - 10)
        if self.spb_start_time.value() < 0:
            self.spb_start_time.setValue(0)

    def combox_electrode(self, index):
        cur = self.comb_electrode.currentText()
        self.cur_electrode_key = cur