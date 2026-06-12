# mmVitals-Projekt

Aditional Tools to install 
-----------------------------------------------------------------------------------------------------------------------
    1. reinstalled the Radar Tool box using the chrome plugin and software tool found here: 
<img width="660" height="370" alt="image" src="https://github.com/user-attachments/assets/deda2cb1-0b21-4a00-b76a-80517c453446" />

    2. Installed mmWave sdk: https://www.ti.com/tool/MMWAVE-SDK#downloads
    
    3. Installed mmWave Studio: https://www.ti.com/tool/MMWAVE-STUDIO
        A myTi Acoutn is required here. 
    4. Install MCR_R2015aSP1_win32_installer: https://www.mathworks.com/products/compiler/matlab-runtime.html

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
    https://www.ti.com/tool/IWR1843BOOST#order-start-development

 mmmWave Studio Configuration for Test 1
-----------------------------------------------------------------------------------------------------------------------   
1. To configure the IWR1843BOOST with mmWave Studio for raw data capture, set the SOP (Sensor On-Chip ootloader) switches to Development Mode: SOP0 ON, SOP1 ON, SOP2 OFF. SOP Switches tell the radar chip what to do when it turns on.
2. Static Configuration: Enable RF LDO Bypass and also select PA LDO I/P Disable
3. For the IWR1843BOOST, you must configure exactly 2 LVDS lanes  using the Format 0 configuraation(Format 1 is not supported by this chip). Hinweis: The IWR1843 hardware only routes 2 data lanes to the DCA1000EVM interface.
4. When configuring the SensorConfi tab, apply the following settings under the LVDS Lane Config section: Lane Format: Select Format 0, Lane 1: Enable, Lane 2: Enable, Lane 3: Leave Disabled, Lane 4: Leave Disabled. Hinweis: The settings you apply in the SensorConfig tab configure the radar chip itself. You must ensure that the data capture card is expecting the same layout. Click on the DCA1000 CLI control sectionn and make sure the capture card lane mode is also explicitly set to 2 Lanes. If this is set to 4 lanes while the sensor outputs 2, your data stream will desynchronize, and mmWave Studio will throw a packet -loss or post-processing error.
5. SensorConfiguration Tab Inputs:
   <img width="1102" height="663" alt="WhatsApp Image 2026-06-11 at 19 41 16" src="https://github.com/user-attachments/assets/0ff0b56e-5746-43a9-8ecb-4841ac96e37f" />
