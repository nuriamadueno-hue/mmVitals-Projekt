# mmVitals-Projekt

Aditional Tools to install 
-----------------------------------------------------------------------------------------------------------------------
    1. reinstalled the Radar Tool box using the chrome plugin and software tool found here: 
<img width="660" height="370" alt="image" src="https://github.com/user-attachments/assets/deda2cb1-0b21-4a00-b76a-80517c453446" />

    2. Installed mmWave sdk: https://www.ti.com/tool/MMWAVE-SDK#downloads
    
    3. Installed mmWave Studio: https://www.ti.com/tool/MMWAVE-STUDIO
        A myTi Acoutn is required here. 

Configure IWR1843BOOST
-----------------------------------------------------------------------------------------------------------------------
    1. Source Code for IWR1843BOOST Radar
        Location: C:\ti\mmwave_sdk_03_06_02_00-LTS\packages\ti\control\mmwavelink
        Flash usin UniFlasch 
    2. Send Config command "CSI2/LVDS Lane Config" to IWR1843BOOST Radar to activate the LVDS output
        Location: 
        send using 
    3. Send Config command "ADC Config" to IWR1843BOOST Radar to set the data structure (e.g., "16-bit, Complex IQ data")
        Location: 
        send using 
     4. Send Config command "Data Path Config" to IWR1843BOOST Radar to set the LVDS Stream Enabled
        Location: 
        send using

Configure DCA1000EVM
-----------------------------------------------------------------------------------------------------------------------
    1. Configure Your PC Network Card
        PC IP Address:   192.168.33.30
        Subnet Mask:     255.255.255.0
        
        Power shell comands to achive this:
            "New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.33.30 -PrefixLength 24"
        or
            "Set-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.33.30 -PrefixLength 24"

        May be verified using cmd Window and the comand "ipconfig"
        
    2. Send the FPGA Configuration Command: Through mmWave Studio (or custom UDP sockets)
        "Capture raw data mode"
        "Split files into 100MB chunks"
        "Stream data packets to target IP 192.168.33.30 on port 4098"

        This can be done using the DCA1000EVM_CLI.exe locateted here: C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc
            to execute this correctly we should create a jason file with the comands we want to send to the FPGA and call the json file cf.json
            then we can navigate the cmd line to the location of the DCA1000EVM_CLI.exe and use the comnand DCA1000EVM_CLI.exe fpga cf.json
    
    3. Arm the DCA1000 send command:
         "Arm Capture"
         Same method as above

    (This can later be autoamted with python lets just move forward as is for testing) 

Items that may be of use later
-----------------------------------------------------------------------------------------------------------------------
    mmWave_Demo_Visualizer: https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/
