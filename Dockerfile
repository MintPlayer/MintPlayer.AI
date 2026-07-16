# MintPlayer.AI.ReinforcementLearning Playground — multi-stage build (PRD §7.7 / PLAN M10).
# Models + the public gallery live on the /data volume, so trained AI survives
# container restarts and upgrades; a fresh volume trains itself at startup.

# --- Stage 1: Angular production bundle ---
# bookworm-slim, not alpine: the polyglot CLI extracted below is a native glibc binary.
FROM node:22-bookworm-slim AS client
# The gitignored *_solver.ts twins are Polyglot build outputs, not sources — they must be
# generated before `ng build`. No dotnet needed for that: the polyglot CLI is a self-contained
# native binary inside the MSBuild nupkg (a plain zip), fetched straight from nuget.org.
# (There is no DotnetTool/dnx package yet; the in-box plugins ship next to the binary.)
ARG POLYGLOT_VERSION=0.8.1
ADD https://api.nuget.org/v3-flatcontainer/mintplayer.polyglot.msbuild/${POLYGLOT_VERSION}/mintplayer.polyglot.msbuild.${POLYGLOT_VERSION}.nupkg /tmp/polyglot.nupkg
RUN apt-get update && apt-get install -y --no-install-recommends unzip \
 && rm -rf /var/lib/apt/lists/* \
 && unzip -q /tmp/polyglot.nupkg -d /opt/polyglot \
 && chmod +x /opt/polyglot/tools/linux-x64/polyglot
WORKDIR /repo/src/RLDemo.Web/ClientApp
COPY src/RLDemo.Web/ClientApp/package*.json ./
RUN npm ci
COPY src/RLDemo.Web/ClientApp/ ./
# Bare `polyglot build` discovers its .pg inputs from pgconfig.json's include patterns;
# --target typescript emits only the TS twins, routed into src/app/<game>/ right here.
COPY pgconfig.json /repo/
COPY src/MintPlayer.AI.ReinforcementLearning.Environments/ /repo/src/MintPlayer.AI.ReinforcementLearning.Environments/
RUN cd /repo && /opt/polyglot/tools/linux-x64/polyglot build --target typescript
RUN npm run build

# --- Stage 2: .NET publish (SPA build skipped — stage 1 already did it) ---
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
# stage above already produced the production bundle).
RUN dotnet publish src/RLDemo.Web/RLDemo.Web.csproj -c Release -o /app/publish \
    -p:SkipAngularPublish=true -p:EnableSpaBuilder=false

# --- Stage 3: runtime ---
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app/publish .
COPY --from=client /repo/src/RLDemo.Web/ClientApp/dist/ClientApp/browser ./ClientApp/dist/ClientApp/browser
# Shipped pre-trained checkpoints: a fresh /data volume seeds itself from these at
# startup, so the container is instantly ready instead of training from scratch.
COPY models/ ./models/
ENV DataDirectory=/data
ENV SeedModelsDirectory=/app/models
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["dotnet", "RLDemo.Web.dll"]
