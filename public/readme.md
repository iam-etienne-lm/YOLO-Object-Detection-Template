# Requirements

Installez de quoi host un site Apache / PHP :

```bash
sudo bash
apt-get update
apt-get install apache2 php libapache2-mod-php
systemctl restart apache2
```

Placez le contenu de ce dossier dans /var/www/html
Puis donnez les autorisations root à ce dossier :

```bash
sudo chmod 777 -R /var/www/html
```

Modifier la configuration de PHP pour accepter l'upload de fichiers plus lours (> 8Mo)

```bash
sudo nano /etc/php/7.4/apache2/php.ini
```

    file_uploads = On
    post_max_size = 800M
    upload_max_filesize = 800M

Installer NodeJS

```bash
sudo apt install nodejs
sudo apt install npm
```
