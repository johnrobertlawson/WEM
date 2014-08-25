"""
Utility scripts to help with directory, file, etc issues
"""

import os
import GIS_tools as utils
try:
    import paramiko
except ImportError:
    print("Module paramiko unavailable.")

def dir_from_naming(root,*args):
    """
    Generate file path from arguments

    Inputs:
    root    :    file path base
    args     :    list of arguments to join as separate folders
    """
    l = [str(a) for a in args]
    path = os.path.join(root,*l)
    return path

def ssh_client(domain,username,password):
    #key = paramiko.RSAKey(data=base64.decodestring('AAA...'))
    client = paramiko.SSHClient()
    #client.get_host_keys().add('ssh.example.com', 'ssh-rsa', key)
    client.connect(domain, username=username, password=password)
    return client
    stdin, stdout, stderr = client.exec_command('ls')
    for line in stdout:
        print '... ' + line.strip('\n')
    client.close()
