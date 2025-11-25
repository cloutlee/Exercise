import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl, QDir
from PyQt5.QtWebChannel import QWebChannel
import numpy as np

class BloodPressureBridge(QObject):
    result_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
    
    @pyqtSlot(str)
    def predict_hypotension_risk(self, json_data):
        try:
            data = json.loads(json_data)
            age = int(data.get('age', 0))
            height = int(data.get('height', 0))


            hypotension_prob = 0.5
            
            result = {
                "hypotension_probability": round(hypotension_prob * 100, 2),
                "risk_level": self.get_risk_level(round(hypotension_prob * 100, 2))
            }
            
            self.result_signal.emit(json.dumps(result, ensure_ascii=False))
            
        except Exception as e:
            error_result = {
                "error": f"計算錯誤: {str(e)}",
                "hypotension_probability": 0
            }
            self.result_signal.emit(json.dumps(error_result, ensure_ascii=False))
    
    def get_risk_level(self, probability):
        if probability < 20:
            return "低風險"
        elif probability < 50:
            return "中等風險"
        else:
            return "高風險"    


class HypotensionPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AAA')
        self.setGeometry(100, 100, 1400, 900)
        
        self.web_view = QWebEngineView()
        self.setCentralWidget(self.web_view)
        
        self.bridge = BloodPressureBridge()
        self.channel = QWebChannel()
        self.channel.registerObject('bpBridge', self.bridge)
        self.web_view.page().setWebChannel(self.channel)
        
        
        html_path = QUrl.fromLocalFile(QDir.currentPath() + "./test.html")
        self.web_view.load(html_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HypotensionPredictionApp()
    window.show()
    sys.exit(app.exec_())
