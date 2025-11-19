import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Image } from "react-native";
const Listitem = ({
  name,
  symbol,
  currentPrice,
  priceChangePercentage7d,
  logoUrl,
  onPress,
}) => {
  const priceChangeColor = priceChangePercentage7d > 0 ? "#10B981" : "#EF4444";
  const isPositive = priceChangePercentage7d > 0;
  
  return (
    <TouchableOpacity 
      onPress={onPress}
      activeOpacity={0.7}
      style={styles.touchable}
    >
      <View style={styles.ItemWrapper}>
        {/* Left side */}
        <View style={styles.leftWrapper}>
          <View style={styles.imageContainer}>
            <Image 
              style={styles.image} 
              source={{ uri: logoUrl }}
              onError={(error) => {
                // Image failed to load, but we'll just show the placeholder
              }}
              resizeMode="contain"
            />
          </View>
          <View style={styles.titlesWrapper}>
            <Text style={styles.title} numberOfLines={1}>{name}</Text>
            <Text style={styles.subtitle}>{symbol.toUpperCase()}</Text>
          </View>
        </View>
        {/* Right side */}
        <View style={styles.rightWrapper}>
          <Text style={styles.price}>
            ${currentPrice.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Text>
          <View style={[styles.changeContainer, { backgroundColor: isPositive ? "#D1FAE5" : "#FEE2E2" }]}>
            <Text style={[styles.changeText, { color: priceChangeColor }]}>
              {isPositive ? "↑" : "↓"} {Math.abs(priceChangePercentage7d).toFixed(2)}%
            </Text>
          </View>
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  touchable: {
    marginHorizontal: 20,
    marginBottom: 12,
  },
  ItemWrapper: {
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderRadius: 16,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
    borderWidth: 1,
    borderColor: "#F3F4F6",
  },
  leftWrapper: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  imageContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#F9FAFB",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
    overflow: "hidden",
  },
  image: {
    height: 40,
    width: 40,
  },
  titlesWrapper: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 13,
    color: "#6B7280",
    fontWeight: "500",
  },
  rightWrapper: {
    alignItems: "flex-end",
    marginLeft: 12,
  },
  price: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    marginBottom: 6,
  },
  changeContainer: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  changeText: {
    fontSize: 12,
    fontWeight: "600",
  },
});
export default Listitem;
