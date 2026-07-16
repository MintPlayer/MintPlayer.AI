# MintPlayer.AI.ReinforcementLearning Playground — multi-stage build (PRD §7.7 / PLAN M10).
# Models + the public gallery live on the /data volume, so trained AI survives
# container restarts and upgrades; a fresh volume trains itself at startup.

# --- Stage 1: .NET publish ---
# Runs FIRST: the Polyglot transpile inside this publish (MintPlayer.Polyglot.MSBuild,
# restored like any package) also emits the gitignored *_solver.ts twins into
# src/RLDemo.Web/ClientApp/src/app/** via pgconfig.json's include rules — the Angular
# stage below copies them from here, so `dotnet build` genuinely precedes `ng build`.
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
# pgconfig.json: required by the Polyglot transpile (0.8.1+) — the CLI refuses to build
# without a config declaring the target set (found by walk-up from the .pg files).
COPY MintPlayer.AI.ReinforcementLearning.sln Directory.Build.props* nuget.config* pgconfig.json ./
COPY src/MintPlayer.AI.ReinforcementLearning.Core/MintPlayer.AI.ReinforcementLearning.Core.csproj src/MintPlayer.AI.ReinforcementLearning.Core/
COPY src/MintPlayer.AI.ReinforcementLearning.Environments/MintPlayer.AI.ReinforcementLearning.Environments.csproj src/MintPlayer.AI.ReinforcementLearning.Environments/
# RLDemo.Web references Ilgpu (resident GPU forward for the self-taught cube solver; ILGPU is pure
# managed and falls back to CPU when the runtime image has no CUDA device) — its csproj must be present
# for restore to resolve the project reference.
COPY src/MintPlayer.AI.ReinforcementLearning.Ilgpu/MintPlayer.AI.ReinforcementLearning.Ilgpu.csproj src/MintPlayer.AI.ReinforcementLearning.Ilgpu/
COPY src/RLDemo.Web/RLDemo.Web.csproj src/RLDemo.Web/
RUN dotnet restore src/RLDemo.Web/RLDemo.Web.csproj
COPY src/ src/
# SkipAngularPublish disables our csproj's npm publish target; EnableSpaBuilder=false
# disables MintPlayer.AspNetCore.NodeServices' automatic npm install/build (the Node
# stage below produces the production bundle).
RUN dotnet publish src/RLDemo.Web/RLDemo.Web.csproj -c Release -o /app/publish \
    -p:SkipAngularPublish=true -p:EnableSpaBuilder=false

# --- Stage 2: Angular production bundle ---
FROM node:22-alpine AS client
WORKDIR /client
COPY src/RLDemo.Web/ClientApp/package*.json ./
RUN npm ci
COPY src/RLDemo.Web/ClientApp/ ./
# Overlay the app sources from the build stage: same files as the context copy above,
# PLUS the generated *_solver.ts twins the publish emitted (gitignored, so absent here).
COPY --from=build /src/src/RLDemo.Web/ClientApp/src/app/ ./src/app/
RUN npm run build

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
