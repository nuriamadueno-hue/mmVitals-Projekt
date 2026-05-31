# mmVitals-Projekt

Aditional Tools to install 
-----------------------------------------------------------------------------------------------------------------------
    1. reinstalled the Radar Tool box using the chrome plugin and software tool found here: 
        <img width="660" height="370" alt="image" src="https://github.com/user-attachments/assets/deda2cb1-0b21-4a00-b76a-80517c453446" />

    2. Installed mmWave Studio: https://www.ti.com/tool/MMWAVE-SDK#downloads


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
    2. Send the FPGA Configuration Command: Through mmWave Studio (or custom UDP sockets)
        "Capture raw data mode"
        "Split files into 100MB chunks"
        "Stream data packets to target IP 192.168.33.30 on port 4098"
     3. Arm the DCA1000 send command:
         "Arm Capture"

Items that may be of use later
-----------------------------------------------------------------------------------------------------------------------
    mmWave_Demo_Visualizer: https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/
