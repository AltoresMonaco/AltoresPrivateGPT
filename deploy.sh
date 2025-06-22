#!/bin/bash

# Script de déploiement PrivateGPT avec Ollama
# Usage: ./deploy.sh

set -e

echo "🚀 Déploiement de PrivateGPT avec Ollama"

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez installer Docker avant de continuer."
    exit 1
fi

# Vérifier que Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Veuillez installer Docker Compose avant de continuer."
    exit 1
fi

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p local_data/private_gpt
mkdir -p models

# Télécharger la dernière image Docker
echo "📥 Téléchargement de l'image Docker..."
docker pull docker.io/altores/privateg-gpt:latest

# Démarrer les services
echo "🔧 Démarrage des services..."
docker-compose -f docker-compose.production.yml up -d

# Attendre que les services soient prêts
echo "⏳ Attente du démarrage des services..."
sleep 10

# Vérifier que Ollama est accessible
echo "🔍 Vérification d'Ollama..."
timeout 30 bash -c 'until curl -f http://localhost:11434/api/version; do sleep 2; done' || {
    echo "❌ Ollama ne répond pas. Vérifiez les logs avec: docker-compose -f docker-compose.production.yml logs ollama"
    exit 1
}

# Télécharger les modèles requis
echo "📦 Téléchargement des modèles Ollama..."
docker exec ollama ollama pull lucifers/Polaris-4B-Preview.Q8_0:latest
docker exec ollama ollama pull nomic-embed-text

# Vérifier que PrivateGPT est accessible
echo "🔍 Vérification de PrivateGPT..."
timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 5; done' || {
    echo "❌ PrivateGPT ne répond pas. Vérifiez les logs avec: docker-compose -f docker-compose.production.yml logs private-gpt"
    exit 1
}

echo "✅ Déploiement terminé avec succès!"
echo "🌐 PrivateGPT est accessible sur: http://localhost:8080"
echo "🔧 Interface d'administration Ollama: http://localhost:11434"
echo ""
echo "📋 Commandes utiles:"
echo "  - Voir les logs: docker-compose -f docker-compose.production.yml logs -f"
echo "  - Arrêter les services: docker-compose -f docker-compose.production.yml down"
echo "  - Redémarrer: docker-compose -f docker-compose.production.yml restart" 