# Guide de Déploiement PrivateGPT

Ce guide explique comment déployer PrivateGPT avec Ollama en utilisant Docker.

## Prérequis

- Docker installé sur le serveur
- Docker Compose installé
- Accès GPU (recommandé pour de meilleures performances)
- Au moins 8GB de RAM disponible
- 20GB d'espace disque libre

## Déploiement Automatique

### Option 1: Script de déploiement automatique

```bash
# Télécharger les fichiers de déploiement
wget https://raw.githubusercontent.com/AltoresMonaco/AltoresPrivateGPT/main/docker-compose.production.yml
wget https://raw.githubusercontent.com/AltoresMonaco/AltoresPrivateGPT/main/deploy.sh

# Rendre le script exécutable et l'exécuter
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Déploiement manuel

```bash
# 1. Créer les répertoires nécessaires
mkdir -p local_data/private_gpt
mkdir -p models

# 2. Télécharger le docker-compose
wget https://raw.githubusercontent.com/AltoresMonaco/AltoresPrivateGPT/main/docker-compose.production.yml

# 3. Démarrer les services
docker-compose -f docker-compose.production.yml up -d

# 4. Télécharger les modèles Ollama
docker exec ollama ollama pull lucifers/Polaris-4B-Preview.Q8_0:latest
docker exec ollama ollama pull nomic-embed-text
```

## Configuration

### Variables d'environnement

Vous pouvez personnaliser le déploiement en modifiant les variables d'environnement dans le `docker-compose.production.yml` :

```yaml
environment:
  - PGPT_PROFILES=ollama  # Profil de configuration à utiliser
  - APP_ENV=prod         # Environnement d'application
```

### Ports

- **8080**: Interface web PrivateGPT
- **11434**: API Ollama

### Volumes

- `./local_data`: Données persistantes de PrivateGPT (base vectorielle, etc.)
- `./models`: Modèles téléchargés
- `ollama_data`: Données Ollama (modèles, configuration)

## Utilisation

Une fois déployé, l'application est accessible sur :
- **Interface web**: http://localhost:8080
- **API**: http://localhost:8080/docs

## Gestion des Services

### Voir les logs
```bash
docker-compose -f docker-compose.production.yml logs -f
```

### Redémarrer les services
```bash
docker-compose -f docker-compose.production.yml restart
```

### Arrêter les services
```bash
docker-compose -f docker-compose.production.yml down
```

### Mettre à jour l'image
```bash
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

## Résolution de Problèmes

### Ollama ne démarre pas
- Vérifiez que le GPU est accessible : `nvidia-smi`
- Vérifiez les logs : `docker-compose -f docker-compose.production.yml logs ollama`

### PrivateGPT ne se connecte pas à Ollama
- Vérifiez que Ollama répond : `curl http://localhost:11434/api/version`
- Vérifiez la configuration réseau Docker

### Modèles non trouvés
- Téléchargez manuellement : `docker exec ollama ollama pull <model-name>`
- Vérifiez l'espace disque disponible

## Personnalisation

Pour personnaliser la configuration, vous pouvez :

1. Modifier les fichiers de configuration dans le volume `./local_data`
2. Créer votre propre `docker-compose.yml` basé sur `docker-compose.production.yml`
3. Utiliser des variables d'environnement pour surcharger les paramètres

## Support

Pour toute question ou problème, consultez :
- Les logs des conteneurs
- La documentation officielle de PrivateGPT
- Les issues GitHub du projet 