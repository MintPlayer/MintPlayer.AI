# RL.NET Playground — multi-stage build (PRD §7.7 / PLAN M10).
# Models + the public gallery live on the /data volume, so trained AI survives
# container restarts and upgrades; a fresh volume trains itself at startup.

# --- Stage 1: Angular production bundle ---
FROM node:22-alpine AS client
WORKDIR /client
COPY src/RL.NET.Web/ClientApp/package*.json ./
RUN npm ci
COPY src/RL.NET.Web/ClientApp/ ./
RUN npm run build

# --- Stage 2: .NET publish (SPA build skipped — stage 1 already did it) ---
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY RL.NET.sln Directory.Build.props* nuget.config* ./
COPY src/RL.NET.Core/RL.NET.Core.csproj src/RL.NET.Core/
COPY src/RL.NET.Environments/RL.NET.Environments.csproj src/RL.NET.Environments/
COPY src/RL.NET.Web/RL.NET.Web.csproj src/RL.NET.Web/
RUN dotnet restore src/RL.NET.Web/RL.NET.Web.csproj
COPY src/ src/
# SkipAngularPublish disables our csproj's npm publish target; EnableSpaBuilder=false
# disables MintPlayer.AspNetCore.NodeServices' automatic npm install/build (the Node
# stage above already produced the production bundle).
RUN dotnet publish src/RL.NET.Web/RL.NET.Web.csproj -c Release -o /app/publish \
    -p:SkipAngularPublish=true -p:EnableSpaBuilder=false

# --- Stage 3: runtime ---
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app/publish .
COPY --from=client /client/dist/ClientApp/browser ./ClientApp/dist/ClientApp/browser
ENV DataDirectory=/data
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["dotnet", "RL.NET.Web.dll"]
