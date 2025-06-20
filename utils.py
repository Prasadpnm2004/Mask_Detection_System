from plyer import notification
import os

def alert_user(message):
    notification.notify(
        title='Face Mask Alert',
        message=message,
        app_icon=None,
        timeout=5
    )
    try:
        os.system('afplay /System/Library/Sounds/Ping.aiff')
        from playsound import playsound
        playsound("alert.wav")
    except:
        pass