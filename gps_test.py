import meshtastic.serial_interface
from pubsub import pub

def on_receive(packet, interface):
    d = packet.get("decoded", {})
    print(packet.get("fromId"), d.get("portnum"), d.get("position"))

pub.subscribe(on_receive, "meshtastic.receive")
iface = meshtastic.serial_interface.SerialInterface()
input("Listening... press Enter to stop\n")
iface.close()