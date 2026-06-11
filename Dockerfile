# MintPlayer.AI.ReinforcementLearning Playground — multi-stage build (PRD §7.7 / PLAN M10).
# Models + the public gallery live on the /data volume, so trained AI survives
# container restarts and upgrades; a fresh volume trains itself at startup.

# --- Stage 1: Angular production bundle ---
FROM node:22-alpine AS client
WORKDIR /client
COPY src/RLDemo.Web/ClientApp/package*.json ./
RUN npm ci
COPY src/RLDemo.Web/ClientApp/ ./
RUN npm run build

# --- Stage 2: .NET publish (SPA build skipped — stage 1 already did it) ---
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY MintPlayer.AI.ReinforcementLearning.sln Directory.Build.props* nuget.config* ./
COPY src/MintPlayer.AI.ReinforcementLearning.Core/MintPlayer.AI.ReinforcementLearning.Core.csproj src/MintPlayer.AI.ReinforcementLearning.Core/
COPY src/MintPlayer.AI.ReinforcementLearning.Environments/MintPlayer.AI.ReinforcementLearning.Environments.csproj src/MintPlayer.AI.ReinforcementLearning.Environments/
COPY src/RLDemo.Web/RLDemo.Web.csproj src/RLDemo.Web/
RUN dotnet restore src/RLDemo.Web/RLDemo.Web.csproj
COPY src/ src/
# SkipAngularPublish disables our csproj's npm publish target; EnableSpaBuilder=false
# disables MintPlayer.AspNetCore.NodeServices' automatic npm install/build (the Node
# stage above already produced the production bundle).
RUN dotnet publish src/RLDemo.Web/RLDemo.Web.csproj -c Release -o /app/publish \
    -p:SkipAngularPublish=true -p:EnableSpaBuilder=false

# --- Stage 3: runtime ---
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app/publish .
COPY --from=client /client/dist/ClientApp/browser ./ClientApp/dist/ClientApp/browser
# Shipped pre-trained checkpoints: a fresh /data volume seeds itself from these at
# startup, so the container is instantly ready instead of training from scratch.
COPY models/ ./models/
ENV DataDirectory=/data
ENV SeedModelsDirectory=/app/models
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["dotnet", "RLDemo.Web.dll"]
