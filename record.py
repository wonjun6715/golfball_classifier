import pyaudio
import wave
import json
import serial
from array import array

FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100 
CHUNK=1024  
RECORD_SECONDS=2
Position = 1

# 아두이노로부터 젯슨나노가 플래그를 받는 코드
connection = serial.Serial(port="COM3", baudrate=9600)
connection.reset_input_buffer()
while True:
    data = connection.readline().decode("utf-8")
    try:
        dict_json = json.loads(data)
        if 'STEP' in dict_json and dict_json['STEP'] == 1:
            audio=pyaudio.PyAudio() #instantiate the pyaudio
            stream=audio.open(format=FORMAT,channels=CHANNELS,input_device_index=1,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)
            #starting recording
            print("start recording")
            frames=[]
            for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
                data=stream.read(CHUNK)
                data_chunk=array('h',data)
                vol=max(data_chunk)
                frames.append(data) 
            print("finish recording")
            #end of recording
            stream.stop_stream()
            stream.close()
            audio.terminate()
            #writing to file
            wavfile=wave.open("NORMAL_{}.wav".format(Position),'wb')
            Position+=1
            wavfile.setnchannels(CHANNELS)
            wavfile.setsampwidth(audio.get_sample_size(FORMAT))
            wavfile.setframerate(RATE)
            wavfile.writeframes(b''.join(frames))#append frames recorded to file
            wavfile.close()
            
            # 모델 연산 코드 추가 필요!!!
            
            
            
            flag_value = int(input()) # 임시로 모델의 결과값을 입력하여 테스트 
            if flag_value == 1: # normal ball
                doc = {"act": flag_value}
            elif flag_value == 2: # broken ball
                doc = {"act": flag_value}
                 
        connection.write(json.dumps(doc).encode('utf-8'))
    except json.JSONDecodeError as e:
        print("JSON:", e)
        
    